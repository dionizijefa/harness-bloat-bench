const POLL_MS = 2000;

const app = {
  data: { runs: [], default_results: [] },
  selectedRunId: null,
  logStream: "stdout",
  refreshing: false,
};

const els = {
  form: document.querySelector("#runForm"),
  syncPill: document.querySelector("#syncPill"),
  refreshButton: document.querySelector("#refreshButton"),
  metrics: document.querySelector("#metrics"),
  runCount: document.querySelector("#runCount"),
  runList: document.querySelector("#runList"),
  runDetail: document.querySelector("#runDetail"),
  cancelButton: document.querySelector("#cancelButton"),
  resultFilter: document.querySelector("#resultFilter"),
  resultLimit: document.querySelector("#resultLimit"),
  resultChart: document.querySelector("#resultChart"),
  resultTable: document.querySelector("#resultTable"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function statusClass(value) {
  return String(value || "unknown").toLowerCase().replace(/[^a-z0-9_]+/g, "_");
}

function statusPill(value) {
  const status = value || "unknown";
  return `<span class="status ${statusClass(status)}">${escapeHtml(status)}</span>`;
}

function normalizePassed(value) {
  if (value === true || value === "true" || value === 1 || value === "1") return true;
  if (value === false || value === "false" || value === 0 || value === "0") return false;
  return null;
}

function formatDuration(seconds) {
  const raw = Number(seconds);
  if (!Number.isFinite(raw) || raw <= 0) return "-";
  if (raw < 60) return `${raw.toFixed(1)}s`;
  const minutes = Math.floor(raw / 60);
  const rest = Math.round(raw % 60).toString().padStart(2, "0");
  if (minutes < 60) return `${minutes}:${rest}`;
  const hours = Math.floor(minutes / 60);
  const min = (minutes % 60).toString().padStart(2, "0");
  return `${hours}:${min}:${rest}`;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortId(value) {
  const text = String(value || "");
  if (text.length <= 22) return text;
  return `${text.slice(0, 13)}...${text.slice(-6)}`;
}

function resourceText(row) {
  const cpus = row.cpus ?? row.resources?.cpus ?? 0;
  const memory = row.memory_mb ?? row.resources?.memory_mb ?? 0;
  const gpus = row.gpus ?? row.resources?.gpus ?? 0;
  const memoryText = Number(memory) ? `${Math.round(Number(memory) / 1024)}GB` : "0GB";
  return `${cpus} CPU, ${memoryText}, ${gpus} GPU`;
}

function allResultRows() {
  const defaultRows = (app.data.default_results || []).map((row) => ({
    ...row,
    _runId: "scheduler",
    _source: "outputs/scheduler",
  }));
  const uiRows = (app.data.runs || []).flatMap((run) =>
    (run.results || []).map((row) => ({
      ...row,
      _runId: run.id,
      _source: run.id,
      _request: run.request || {},
    })),
  );
  return [...uiRows, ...defaultRows];
}

function runCounts(run) {
  const counts = run.state?.counts;
  if (counts && Number.isFinite(Number(counts.total))) return counts;
  const total = (run.results || []).length;
  return {
    total,
    finished: total,
    running: run.status === "running" ? 1 : 0,
    pending: 0,
    passed: (run.results || []).filter((row) => row.status === "passed").length,
    failed: (run.results || []).filter((row) => row.status === "failed").length,
    skipped: (run.results || []).filter((row) => row.status === "skipped").length,
    dry_run: (run.results || []).filter((row) => row.status === "dry_run").length,
  };
}

function percentage(numerator, denominator) {
  const top = Number(numerator);
  const bottom = Number(denominator);
  if (!Number.isFinite(top) || !Number.isFinite(bottom) || bottom <= 0) return 0;
  return Math.max(0, Math.min(100, (top / bottom) * 100));
}

function selectedRun() {
  return (app.data.runs || []).find((run) => run.id === app.selectedRunId) || null;
}

function renderMetrics() {
  const rows = allResultRows();
  const known = rows.filter((row) => normalizePassed(row.passed) !== null);
  const passed = known.filter((row) => normalizePassed(row.passed) === true).length;
  const failed = rows.filter((row) => row.status === "failed" || normalizePassed(row.passed) === false).length;
  const active = (app.data.runs || []).filter((run) => run.status === "running" || run.status === "canceling").length;
  const runtimes = rows.map((row) => Number(row.runtime_seconds)).filter((value) => Number.isFinite(value) && value > 0);
  const avgRuntime = runtimes.length
    ? runtimes.reduce((sum, value) => sum + value, 0) / runtimes.length
    : 0;
  const passRate = known.length ? Math.round((passed / known.length) * 100) : 0;

  els.metrics.innerHTML = [
    ["Active", active],
    ["Rows", rows.length],
    ["Pass rate", known.length ? `${passRate}%` : "-"],
    ["Failed", failed],
    ["Avg runtime", formatDuration(avgRuntime)],
  ]
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderRuns() {
  const runs = app.data.runs || [];
  if (!app.selectedRunId && runs.length) app.selectedRunId = runs[0].id;
  if (app.selectedRunId && !runs.some((run) => run.id === app.selectedRunId)) {
    app.selectedRunId = runs[0]?.id || null;
  }

  els.runCount.textContent = String(runs.length);
  if (!runs.length) {
    els.runList.innerHTML = `<div class="empty">No UI runs yet.</div>`;
    return;
  }

  els.runList.innerHTML = runs
    .map((run) => {
      const counts = runCounts(run);
      const total = Number(counts.total) || 0;
      const finished = Number(counts.finished) || 0;
      const width = percentage(finished, total || (run.status === "complete" ? 1 : 0));
      const request = run.request || {};
      const model = request.model || "default model";
      const include = request.include ? `, ${request.include}` : "";
      const selected = run.id === app.selectedRunId ? " is-selected" : "";
      return `
        <button class="run-row${selected}" type="button" data-run-id="${escapeHtml(run.id)}">
          <div class="row-top">
            <span class="run-id">${escapeHtml(shortId(run.id))}</span>
            ${statusPill(run.status)}
          </div>
          <div class="row-meta">
            <span>${escapeHtml(model)}${escapeHtml(include)}</span>
            <span>${escapeHtml(formatDate(run.created_at))}</span>
          </div>
          <div class="progress-track">
            <span class="progress-fill" style="width:${width}%"></span>
          </div>
          <div class="row-meta">
            <span>${finished}/${total || "?"} finished</span>
            <span>${Number(counts.running) || 0} running, ${Number(counts.pending) || 0} pending</span>
          </div>
        </button>
      `;
    })
    .join("");
}

function resourceBlock(label, used, capacity) {
  const value = Number(used) || 0;
  const max = Number(capacity) || 0;
  const pct = percentage(value, max);
  const tone = pct > 92 ? " danger" : pct > 72 ? " warning" : "";
  const text = max ? `${value}/${max}` : `${value}/0`;
  return `
    <div class="resource">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(text)}</strong>
      <div class="usage-track"><span class="usage-fill${tone}" style="width:${pct}%"></span></div>
    </div>
  `;
}

function renderDetail() {
  const run = selectedRun();
  const canCancel = run && (run.status === "running" || run.status === "canceling");
  els.cancelButton.disabled = !canCancel;

  if (!run) {
    els.runDetail.innerHTML = `<div class="empty">Select a run.</div>`;
    return;
  }

  const state = run.state || {};
  const capacity = state.capacity || {};
  const used = state.used || {};
  const tasks = Array.isArray(state.tasks) ? state.tasks : [];
  const order = { running: 0, pending: 1, failed: 2, passed: 3, dry_run: 4, skipped: 5 };
  const sortedTasks = [...tasks].sort((a, b) => {
    const left = order[a.status] ?? 9;
    const right = order[b.status] ?? 9;
    return left - right || String(a.task_name).localeCompare(String(b.task_name));
  });
  const logText = app.logStream === "stderr" ? run.stderr_tail : run.stdout_tail;

  els.runDetail.innerHTML = `
    <div class="resource-grid">
      ${resourceBlock("CPU", used.cpus, capacity.cpus)}
      ${resourceBlock("Memory MB", used.memory_mb, capacity.memory_mb)}
      ${resourceBlock("Storage MB", used.storage_mb, capacity.storage_mb)}
      ${resourceBlock("GPU", used.gpus, capacity.gpus)}
    </div>

    <div class="task-list">
      ${
        sortedTasks.length
          ? sortedTasks
              .map((task) => {
                const name = task.task_name || "task";
                const meta = [
                  `attempt ${task.attempt || 0}`,
                  formatDuration(task.runtime_seconds),
                  resourceText(task),
                ]
                  .filter(Boolean)
                  .join(" | ");
                return `
                  <div class="task-row">
                    <div class="task-top">
                      <span class="task-name">${escapeHtml(name)}</span>
                      ${statusPill(task.status)}
                    </div>
                    <div class="task-meta">${escapeHtml(meta)}</div>
                  </div>
                `;
              })
              .join("")
          : `<div class="empty">No scheduler state yet.</div>`
      }
    </div>

    <div class="log-box">
      <div class="log-tabs">
        <button type="button" data-log-stream="stdout" class="${app.logStream === "stdout" ? "is-active" : ""}">Stdout</button>
        <button type="button" data-log-stream="stderr" class="${app.logStream === "stderr" ? "is-active" : ""}">Stderr</button>
      </div>
      <pre>${escapeHtml(logText || "")}</pre>
    </div>
  `;
}

function filteredRows() {
  const query = els.resultFilter.value.trim().toLowerCase();
  const limit = Number(els.resultLimit.value) || 100;
  const rows = allResultRows().sort((a, b) => {
    const left = Date.parse(a.finished_at || a.started_at || "") || 0;
    const right = Date.parse(b.finished_at || b.started_at || "") || 0;
    return right - left;
  });
  const filtered = query
    ? rows.filter((row) =>
        [
          row.task_name,
          row.task_id,
          row.status,
          row.model,
          row._runId,
          row._source,
          row.error,
        ]
          .join(" ")
          .toLowerCase()
          .includes(query),
      )
    : rows;
  return filtered.slice(0, limit);
}

function renderChart(rows) {
  const groups = new Map();
  for (const row of rows) {
    const task = row.task_name || row.task_id || "unknown";
    const passed = normalizePassed(row.passed);
    if (!groups.has(task)) groups.set(task, { total: 0, known: 0, passed: 0 });
    const group = groups.get(task);
    group.total += 1;
    if (passed !== null) {
      group.known += 1;
      if (passed) group.passed += 1;
    }
  }

  const items = [...groups.entries()]
    .sort((a, b) => b[1].total - a[1].total || a[0].localeCompare(b[0]))
    .slice(0, 8);

  if (!items.length) {
    els.resultChart.innerHTML = `<div class="empty">No results yet.</div>`;
    return;
  }

  els.resultChart.innerHTML = items
    .map(([task, group]) => {
      const pct = group.known ? Math.round((group.passed / group.known) * 100) : 0;
      const value = group.known ? `${group.passed}/${group.known}` : `${group.total} rows`;
      return `
        <div class="chart-row">
          <span class="chart-label">${escapeHtml(task)}</span>
          <div class="chart-track"><span class="chart-fill" style="width:${pct}%"></span></div>
          <span class="chart-value">${escapeHtml(value)}</span>
        </div>
      `;
    })
    .join("");
}

function renderResults() {
  const rows = filteredRows();
  renderChart(rows);

  if (!rows.length) {
    els.resultTable.innerHTML = `<tr><td colspan="6" class="muted">No matching rows.</td></tr>`;
    return;
  }

  els.resultTable.innerHTML = rows
    .map((row) => {
      const task = row.task_name || row.task_id || "unknown";
      const run = row._runId || row.scheduler_run_id || row.run_id || "scheduler";
      const passed = normalizePassed(row.passed);
      const passedText = passed === null ? "-" : passed ? "yes" : "no";
      const status = row.status || (passed === true ? "passed" : passed === false ? "failed" : "unknown");
      return `
        <tr>
          <td>${escapeHtml(task)}</td>
          <td class="mono">${escapeHtml(shortId(run))}</td>
          <td>${statusPill(status)}</td>
          <td>${escapeHtml(passedText)}</td>
          <td class="runtime">${escapeHtml(formatDuration(row.runtime_seconds))}</td>
          <td class="resources-text">${escapeHtml(resourceText(row))}</td>
        </tr>
      `;
    })
    .join("");
}

function render() {
  renderMetrics();
  renderRuns();
  renderDetail();
  renderResults();
}

function collectFormPayload() {
  const formData = new FormData(els.form);
  const payload = Object.fromEntries(formData.entries());
  for (const key of ["dryRun", "fetch", "openrouter", "yes", "overwriteDataset", "quiet"]) {
    payload[key] = formData.has(key);
  }
  return payload;
}

async function refresh({ quiet = false } = {}) {
  if (app.refreshing) return;
  app.refreshing = true;
  if (!quiet) els.syncPill.textContent = "Refreshing";
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    app.data = await response.json();
    els.syncPill.textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
    render();
  } catch (error) {
    els.syncPill.textContent = "Offline";
    console.error(error);
  } finally {
    app.refreshing = false;
  }
}

async function startRun(event) {
  event.preventDefault();
  const button = els.form.querySelector("button[type='submit']");
  button.disabled = true;
  els.syncPill.textContent = "Starting";
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectFormPayload()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    app.selectedRunId = data.id;
    await refresh({ quiet: true });
  } catch (error) {
    els.syncPill.textContent = "Start failed";
    console.error(error);
    alert(error.message || String(error));
  } finally {
    button.disabled = false;
  }
}

async function cancelSelectedRun() {
  const run = selectedRun();
  if (!run) return;
  els.cancelButton.disabled = true;
  try {
    await fetch(`/api/runs/${encodeURIComponent(run.id)}/cancel`, { method: "POST" });
    await refresh({ quiet: true });
  } catch (error) {
    console.error(error);
  }
}

els.form.addEventListener("submit", startRun);
els.refreshButton.addEventListener("click", () => refresh());
els.cancelButton.addEventListener("click", cancelSelectedRun);
els.resultFilter.addEventListener("input", renderResults);
els.resultLimit.addEventListener("change", renderResults);
els.runList.addEventListener("click", (event) => {
  const row = event.target.closest("[data-run-id]");
  if (!row) return;
  app.selectedRunId = row.dataset.runId;
  render();
});
els.runDetail.addEventListener("click", (event) => {
  const button = event.target.closest("[data-log-stream]");
  if (!button) return;
  app.logStream = button.dataset.logStream;
  renderDetail();
});

refresh();
setInterval(() => refresh({ quiet: true }), POLL_MS);
