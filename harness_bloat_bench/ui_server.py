"""Local web UI for spawning and monitoring scheduler runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "ui_static"
SCHEDULER_SCRIPT = REPO_ROOT / "schedule_harbor_tasks.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ui_runs"
DEFAULT_SCHEDULER_RESULTS = REPO_ROOT / "outputs" / "scheduler" / "results.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compact_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = [str(item) for item in value]
    elif value is None:
        raw_values = []
    else:
        raw_values = [str(value)]

    values: list[str] = []
    for raw in raw_values:
        raw = raw.replace("\n", ",")
        values.extend(part.strip() for part in raw.split(",") if part.strip())
    return values


def text_value(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return str(value).strip() if value is not None else ""


def bool_value(payload: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def add_text_arg(
    command: list[str], payload: dict[str, Any], key: str, flag: str
) -> None:
    value = text_value(payload, key)
    if value:
        command.extend([flag, value])


def build_scheduler_command(payload: dict[str, Any], run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(SCHEDULER_SCRIPT),
        "--output-dir",
        str(run_dir),
        "--jobs-dir",
        str(run_dir / "jobs"),
        "--logs-dir",
        str(run_dir / "logs"),
        "--scheduler-config-dir",
        str(run_dir / "configs"),
        "--results-json",
        str(run_dir / "results.json"),
        "--results-csv",
        str(run_dir / "results.csv"),
        "--state-json",
        str(run_dir / "state.json"),
    ]

    for key, flag in [
        ("dataset", "--dataset"),
        ("datasetPath", "--dataset-path"),
        ("downloadDir", "--download-dir"),
        ("nTasks", "--n-tasks"),
        ("totalCpus", "--total-cpus"),
        ("totalMemory", "--total-memory"),
        ("totalStorage", "--total-storage"),
        ("totalGpus", "--total-gpus"),
        ("reserveCpus", "--reserve-cpus"),
        ("reserveMemory", "--reserve-memory"),
        ("reserveGpus", "--reserve-gpus"),
        ("maxRetries", "--max-retries"),
        ("harborCommand", "--harbor-command"),
        ("harborExtraArgs", "--harbor-extra-args"),
        ("agent", "--agent"),
        ("agentImportPath", "--agent-import-path"),
        ("model", "--model"),
        ("nAttempts", "--n-attempts"),
        ("cpuMode", "--cpu-mode"),
        ("memoryMode", "--memory-mode"),
    ]:
        add_text_arg(command, payload, key, flag)

    for value in split_values(payload.get("include")):
        command.extend(["--include", value])
    for value in split_values(payload.get("exclude")):
        command.extend(["--exclude", value])
    for value in split_values(payload.get("agentEnv")):
        command.extend(["--agent-env", value])
    for value in split_values(payload.get("verifierEnv")):
        command.extend(["--verifier-env", value])
    for value in split_values(payload.get("agentKwarg")):
        command.extend(["--agent-kwarg", value])

    if bool_value(payload, "dryRun"):
        command.append("--dry-run")
    if not bool_value(payload, "fetch", default=True):
        command.append("--no-fetch")
    if bool_value(payload, "overwriteDataset"):
        command.append("--overwrite-dataset")
    if bool_value(payload, "yes", default=True):
        command.append("--yes")
    if bool_value(payload, "quiet"):
        command.append("--quiet")
    if not bool_value(payload, "openrouter", default=True):
        command.append("--no-openrouter")

    return command


def tail_text(path: Path, *, lines: int = 120, max_bytes: int = 65536) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def status_from_returncode(returncode: int | None) -> str:
    if returncode is None:
        return "running"
    if returncode == 0:
        return "complete"
    if returncode < 0:
        return "canceled"
    return "failed"


class RunStore:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def spawn(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = "%s-%s" % (compact_stamp(), uuid.uuid4().hex[:8])
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        command = build_scheduler_command(payload, run_dir)
        stdout_log = run_dir / "scheduler.stdout.log"
        stderr_log = run_dir / "scheduler.stderr.log"
        metadata_path = run_dir / "metadata.json"
        metadata = {
            "id": run_id,
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "run_dir": str(run_dir),
            "command": command,
            "request": payload,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "state_json": str(run_dir / "state.json"),
            "results_json": str(run_dir / "results.json"),
            "results_csv": str(run_dir / "results.csv"),
        }
        write_json(metadata_path, metadata)

        with stdout_log.open("ab") as out, stderr_log.open("ab") as err:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )

        with self._lock:
            self._processes[run_id] = process
        thread = threading.Thread(
            target=self._wait_for_run, args=(run_id,), daemon=True
        )
        thread.start()
        return self.run_snapshot(run_id)

    def _wait_for_run(self, run_id: str) -> None:
        with self._lock:
            process = self._processes.get(run_id)
        if process is None:
            return
        returncode = process.wait()
        metadata_path = self.output_root / run_id / "metadata.json"
        metadata = read_json(metadata_path, {})
        metadata["returncode"] = returncode
        metadata["status"] = status_from_returncode(returncode)
        metadata["finished_at"] = utc_now()
        metadata["updated_at"] = utc_now()
        write_json(metadata_path, metadata)

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            process = self._processes.get(run_id)
        if process is None or process.poll() is not None:
            return self.run_snapshot(run_id)

        metadata_path = self.output_root / run_id / "metadata.json"
        metadata = read_json(metadata_path, {})
        metadata["status"] = "canceling"
        metadata["updated_at"] = utc_now()
        write_json(metadata_path, metadata)

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            process.terminate()
        return self.run_snapshot(run_id)

    def run_dirs(self) -> list[Path]:
        return sorted(
            [path for path in self.output_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def run_snapshot(self, run_id: str) -> dict[str, Any]:
        run_dir = self.output_root / run_id
        metadata = read_json(run_dir / "metadata.json", {})
        state = read_json(run_dir / "state.json", {})
        results = read_json(run_dir / "results.json", [])
        if not isinstance(results, list):
            results = []

        with self._lock:
            process = self._processes.get(run_id)
        returncode = (
            process.poll() if process is not None else metadata.get("returncode")
        )
        is_running = process is not None and returncode is None
        status = (
            "running"
            if is_running
            else metadata.get("status") or state.get("status") or "unknown"
        )
        if not is_running and isinstance(returncode, int):
            status = status_from_returncode(returncode)

        return {
            "id": run_id,
            "status": status,
            "returncode": returncode,
            "created_at": metadata.get("created_at"),
            "finished_at": metadata.get("finished_at"),
            "run_dir": str(run_dir),
            "command": metadata.get("command") or [],
            "request": metadata.get("request") or {},
            "stdout_log": metadata.get("stdout_log")
            or str(run_dir / "scheduler.stdout.log"),
            "stderr_log": metadata.get("stderr_log")
            or str(run_dir / "scheduler.stderr.log"),
            "state": state if isinstance(state, dict) else {},
            "results": results,
            "stdout_tail": tail_text(run_dir / "scheduler.stdout.log", lines=40),
            "stderr_tail": tail_text(run_dir / "scheduler.stderr.log", lines=40),
        }

    def snapshot(self) -> dict[str, Any]:
        runs = [self.run_snapshot(path.name) for path in self.run_dirs()]
        default_results = read_json(DEFAULT_SCHEDULER_RESULTS, [])
        if not isinstance(default_results, list):
            default_results = []
        return {
            "now": utc_now(),
            "runs": runs,
            "default_results": default_results,
            "default_results_path": str(DEFAULT_SCHEDULER_RESULTS),
        }


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length > 1_000_000:
        raise ValueError("request body too large")
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON body") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def make_handler(store: RunStore) -> type[BaseHTTPRequestHandler]:
    class UIHandler(BaseHTTPRequestHandler):
        server_version = "HarnessBloatBenchUI/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/state":
                self.send_json(store.snapshot())
                return

            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/logs"):
                run_id = unquote(parsed.path.split("/")[3])
                query = parse_qs(parsed.query)
                stream = query.get("stream", ["stdout"])[0]
                lines = int(query.get("lines", ["200"])[0])
                snapshot = store.run_snapshot(run_id)
                path = Path(
                    snapshot["stderr_log"]
                    if stream == "stderr"
                    else snapshot["stdout_log"]
                )
                self.send_json(
                    {
                        "run_id": run_id,
                        "stream": stream,
                        "text": tail_text(path, lines=lines),
                    }
                )
                return

            self.serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/runs":
                    payload = parse_json_body(self)
                    self.send_json(store.spawn(payload), status=HTTPStatus.CREATED)
                    return
                if parsed.path.startswith("/api/runs/") and parsed.path.endswith(
                    "/cancel"
                ):
                    run_id = unquote(parsed.path.split("/")[3])
                    self.send_json(store.cancel(run_id))
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - defensive API boundary
                self.send_json(
                    {"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR
                )

        def serve_static(self, request_path: str) -> None:
            if request_path in {"", "/"}:
                request_path = "/index.html"
            relative = request_path.lstrip("/")
            if relative not in {"index.html", "app.css", "app.js"}:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return

            path = STATIC_DIR / relative
            try:
                data = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return

            content_types = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
            }
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type",
                content_types.get(path.suffix, "application/octet-stream"),
            )
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, data: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = json.dumps(data, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            timestamp = dt.datetime.now().strftime("%H:%M:%S")
            sys.stderr.write("[%s] %s\n" % (timestamp, format % args))

    return UIHandler


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the harness-bloat-bench web UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    store = RunStore(args.output_root.expanduser().resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = "http://%s:%d" % (args.host, args.port)
    print("Serving harness-bloat-bench UI at %s" % url, flush=True)
    print("Run output root: %s" % store.output_root, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
