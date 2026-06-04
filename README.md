# harness-bloat-bench

Minimal runner for Codex CLI harness comparisons on Terminal-Bench 2.1 through
Harbor.

The repo is configured for OpenRouter-backed Codex CLI runs and requires:

```sh
export OPENROUTER_API_KEY=sk-or-...
```

Example Qwen3.7 Max run:

```sh
uv run python -m harness_bloat_bench.cli.run_matrix \
  --harness codex-cli \
  --harness-versions latest \
  --models qwen/qwen3.7-max \
  --task-ids adaptive-rejection-sampler \
  --n-attempts 5
```

By default this runs:

```sh
harbor run -d terminal-bench/terminal-bench-2-1
```

Use `--task-ids adaptive-rejection-sampler` to run a subset. Short
Terminal-Bench task IDs are expanded to Harbor task names such as
`terminal-bench/adaptive-rejection-sampler`.

Smoke verification:

```sh
scripts/run_smoke_task.sh
```

This runs a dry-run for `crack-7z-hash`, the shortest
`terminal-bench/terminal-bench-2-1` task by instruction length in the downloaded
Harbor task set. To run it live through OpenRouter:

```sh
scripts/run_smoke_task.sh --live
```

OpenRouter is enabled by default. The runner maps `OPENROUTER_API_KEY` to
`OPENAI_API_KEY`, sets `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, and uses
a local Harbor Codex agent shim so the full OpenRouter model id
`qwen/qwen3.7-max` is passed to `codex exec`.

Results are exported after every Harbor job to:

- `outputs/results.json`
- `outputs/results.jsonl`
- `outputs/results.csv`

Each row is one Terminal-Bench trial/task attempt and includes task id, pass
status, runtime, log paths, input tokens, cached input tokens, output tokens,
reasoning tokens when reported, total tokens, estimated or reported cost, model,
harness version, Harbor run id, timing fields, completion state, task version,
task domain, optional harness event JSON, and the exact command. These files are
intended to be directly loadable from pandas or plotting scripts.

Use `--experiment-id` to attach a stable experiment identifier to every exported
row. If omitted, the generated run id is used.

Use `--dry-run` to verify the Harbor command and output schema without running
Terminal-Bench or requiring an API key.

## Project layout

- `harness_bloat_bench/cli/` contains runnable CLI modules.
- `harness_bloat_bench/ui/` contains the local dashboard server and static
  assets.
- `harness_bloat_bench/openrouter_codex.py` contains the Harbor Codex agent
  shim shared by CLI workflows.
- `scripts/` contains convenience shell wrappers for common smoke runs.
- `tasks/` contains local smoke task fixtures.

The legacy root commands `run_matrix.py`, `schedule_harbor_tasks.py`, and
`harness_bloat_bench.ui_server` remain as compatibility wrappers, but new usage
should prefer the package module paths shown below.

## Resource-aware task scheduling

`harness_bloat_bench.cli.schedule_harbor_tasks` runs selected Terminal-Bench 2 /
Harbor tasks through a lightweight host resource scheduler. It discovers local
`task.toml` files, or downloads a Harbor dataset first, extracts `[environment]` and
`[verifier.environment]` resource settings, and starts one isolated Harbor job per
admitted task with `--n-concurrent 1`.

Dry-run a subset:

```sh
uv run python -m harness_bloat_bench.cli.schedule_harbor_tasks \
  --dataset terminal-bench/terminal-bench-2-1 \
  --include 'terminal-bench/crack-7z-hash' \
  --model qwen/qwen3.7-max \
  --total-cpus 64 \
  --total-memory 128GB \
  --reserve-cpus 4 \
  --reserve-memory 16GB \
  --dry-run
```

Live runs use Harbor CPU and memory enforcement via `--cpus limit`,
`--memory limit`, `--override-cpus`, and `--override-memory`. The scheduler also
passes thread-limiting variables into the task environment and verifier:
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `RAYON_NUM_THREADS`, and `MAKEFLAGS`.

OpenRouter is enabled by default, matching the matrix runner; the scheduler uses
`harness_bloat_bench.openrouter_codex:OpenRouterCodex` automatically for the
default Codex agent.

Scheduler results are written by default to:

- `outputs/scheduler/results.json`
- `outputs/scheduler/results.csv`

The scheduler also writes live state to `outputs/scheduler/state.json`. This
tracks selected, pending, running, skipped, and completed tasks for UI polling.
Scheduler result rows include the same core tracking fields: run id,
experiment id, task id/version/domain, model and harness version, started and
completed timestamps, completion state, token counts, and optional harness
events.

## Web UI

Start the local dashboard:

```sh
uv run python -m harness_bloat_bench.ui.server --host 127.0.0.1 --port 8765
```

Then open <http://127.0.0.1:8765>. The UI spawns
`harness_bloat_bench.cli.schedule_harbor_tasks` runs in isolated directories under
`outputs/ui_runs/`, polls each run's `state.json`, and displays scheduler logs,
resource usage, progress, and result rows as tasks finish.
