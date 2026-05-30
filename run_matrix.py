#!/usr/bin/env python3
"""
Run Terminal-Bench experiments across harness versions and models.

The code is intentionally small and explicit. To add another harness, add a
Harness implementation near CodexCliHarness and register it in get_harness().
"""

import argparse
import csv
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


RESULT_COLUMNS = [
    "run_id",
    "tb_run_id",
    "harness_name",
    "harness_version",
    "model",
    "task_subset",
    "task_id",
    "repeat",
    "passed",
    "runtime_seconds",
    "stdout_log",
    "stderr_log",
    "task_log_dir",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "timestamp",
    "git_commit",
    "terminal_bench_commit",
    "exit_code",
    "output_dir",
    "command",
    "error",
]


@dataclass
class CommandResult:
    exit_code: int
    runtime_seconds: float
    stdout_log: Path
    stderr_log: Path


@dataclass
class RunContext:
    run_id: str
    tb_run_id: str
    harness_name: str
    harness_version: str
    model: str
    task_subset: str
    repeat: int
    output_root: Path
    run_dir: Path
    stdout_log: Path
    stderr_log: Path
    command: List[str]
    git_commit: Optional[str]


class Harness:
    name = ""

    def terminal_bench_args(self, version: str, model: str) -> List[str]:
        raise NotImplementedError


class CodexCliHarness(Harness):
    """Terminal-Bench built-in Codex installed agent."""

    name = "codex-cli"

    def terminal_bench_args(self, version: str, model: str) -> List[str]:
        # Terminal-Bench calls the agent "codex" and uses this kwarg in the
        # container install template: npm install -g @openai/codex@{{ version }}.
        return [
            "--agent",
            "codex",
            "--model",
            model,
            "--agent-kwarg",
            "version=%s" % version,
        ]


def get_harness(name: str) -> Harness:
    normalized = name.strip().lower()
    if normalized == CodexCliHarness.name:
        return CodexCliHarness()
    raise ValueError("unknown harness %r. Known harnesses: codex-cli" % name)


def split_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._-") or "value"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compact_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit(cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def dataset_spec(task_subset: str, dataset_version: str) -> str:
    if "==" in task_subset:
        return task_subset
    return "%s==%s" % (task_subset, dataset_version)


def build_tb_command(
    args: argparse.Namespace,
    harness: Harness,
    harness_version: str,
    model: str,
    task_subset: str,
    tb_run_id: str,
) -> List[str]:
    command = shlex.split(args.tb_command) + ["run"]

    if args.dataset_path:
        command.extend(["--dataset-path", args.dataset_path])
    else:
        command.extend(["--dataset", dataset_spec(task_subset, args.dataset_version)])

    command.extend(harness.terminal_bench_args(harness_version, model))
    command.extend(["--output-path", str(args.tbench_runs_dir)])
    command.extend(["--run-id", tb_run_id])
    command.extend(["--n-concurrent", str(args.n_concurrent)])

    for task_id in split_csv(args.task_ids):
        command.extend(["--task-id", task_id])

    if args.tb_extra_args:
        command.extend(shlex.split(args.tb_extra_args))

    return command


def run_command(command: Sequence[str], stdout_log: Path, stderr_log: Path) -> CommandResult:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open(
        "w", encoding="utf-8"
    ) as err:
        try:
            completed = subprocess.run(
                list(command),
                stdout=out,
                stderr=err,
                text=True,
                check=False,
            )
            exit_code = completed.returncode
        except FileNotFoundError as exc:
            err.write("%s\n" % exc)
            exit_code = 127

    return CommandResult(
        exit_code=exit_code,
        runtime_seconds=time.perf_counter() - start,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )


def load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def parse_timestamp(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def seconds_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    start_dt = parse_timestamp(start)
    end_dt = parse_timestamp(end)
    if not start_dt or not end_dt:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())


def first_present(data: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_pricing(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    data = load_json(Path(path))
    if not isinstance(data, dict):
        raise ValueError("pricing file must be a JSON object")
    return data  # type: ignore[return-value]


def estimate_cost_usd(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    pricing: Dict[str, Dict[str, float]],
) -> Optional[float]:
    price = pricing.get(model) or pricing.get(model.split("/")[-1])
    if not price:
        return None

    if "total_per_million" in price and total_tokens is not None:
        return total_tokens / 1_000_000.0 * float(price["total_per_million"])

    cost = 0.0
    used = False
    if input_tokens is not None and "input_per_million" in price:
        cost += input_tokens / 1_000_000.0 * float(price["input_per_million"])
        used = True
    if output_tokens is not None and "output_per_million" in price:
        cost += output_tokens / 1_000_000.0 * float(price["output_per_million"])
        used = True
    return cost if used else None


def terminal_bench_commit(run_dir: Path) -> Optional[str]:
    metadata = load_json(run_dir / "run_metadata.json")
    if not isinstance(metadata, dict):
        return None
    commit = metadata.get("commit_hash")
    return str(commit) if commit else None


def result_runtime_seconds(result: Dict[str, Any]) -> Optional[float]:
    for start_key, end_key in [
        ("trial_started_at", "trial_ended_at"),
        ("agent_started_at", "agent_ended_at"),
        ("test_started_at", "test_ended_at"),
    ]:
        seconds = seconds_between(result.get(start_key), result.get(end_key))
        if seconds is not None:
            return seconds
    return None


def trial_log_dir(run_dir: Path, result: Dict[str, Any]) -> Optional[str]:
    task_id = result.get("task_id")
    trial_name = result.get("trial_name")
    if task_id and trial_name:
        candidate = run_dir / str(task_id) / str(trial_name)
        if candidate.exists():
            return str(candidate)
    if task_id:
        candidate = run_dir / str(task_id)
        if candidate.exists():
            return str(candidate)
    return None


def parse_aggregate_results(
    context: RunContext,
    command_result: CommandResult,
    pricing: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    results_path = context.run_dir / "results.json"
    data = load_json(results_path)
    if not isinstance(data, dict):
        return []

    results = data.get("results")
    if not isinstance(results, list):
        return []

    rows = []
    tb_commit = terminal_bench_commit(context.run_dir)
    for result in results:
        if not isinstance(result, dict):
            continue

        input_tokens = as_int(
            first_present(result, ["total_input_tokens", "input_tokens", "n_input_tokens"])
        )
        output_tokens = as_int(
            first_present(result, ["total_output_tokens", "output_tokens", "n_output_tokens"])
        )
        total_tokens = as_int(first_present(result, ["total_tokens", "n_total_tokens"]))
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        runtime_seconds = result_runtime_seconds(result)
        if runtime_seconds is None:
            runtime_seconds = command_result.runtime_seconds

        estimated_cost = estimate_cost_usd(
            context.model, input_tokens, output_tokens, total_tokens, pricing
        )

        row = base_row(context, command_result, tb_commit, error="")
        row.update(
            {
                "task_id": result.get("task_id"),
                "passed": result.get("is_resolved"),
                "runtime_seconds": runtime_seconds,
                "task_log_dir": trial_log_dir(context.run_dir, result),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
            }
        )
        rows.append(row)
    return rows


def parse_trial_result_files(
    context: RunContext,
    command_result: CommandResult,
    pricing: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows = []
    tb_commit = terminal_bench_commit(context.run_dir)
    for path in sorted(context.run_dir.glob("*/*/results.json")):
        data = load_json(path)
        if not isinstance(data, dict):
            continue

        input_tokens = as_int(
            first_present(data, ["total_input_tokens", "input_tokens", "n_input_tokens"])
        )
        output_tokens = as_int(
            first_present(data, ["total_output_tokens", "output_tokens", "n_output_tokens"])
        )
        total_tokens = as_int(first_present(data, ["total_tokens", "n_total_tokens"]))
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        estimated_cost = estimate_cost_usd(
            context.model, input_tokens, output_tokens, total_tokens, pricing
        )

        row = base_row(context, command_result, tb_commit, error="")
        row.update(
            {
                "task_id": data.get("task_id") or path.parent.parent.name,
                "passed": data.get("is_resolved"),
                "runtime_seconds": result_runtime_seconds(data)
                or command_result.runtime_seconds,
                "task_log_dir": str(path.parent),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
            }
        )
        rows.append(row)
    return rows


def base_row(
    context: RunContext,
    command_result: CommandResult,
    tb_commit: Optional[str],
    error: str,
) -> Dict[str, Any]:
    return {
        "run_id": context.run_id,
        "tb_run_id": context.tb_run_id,
        "harness_name": context.harness_name,
        "harness_version": context.harness_version,
        "model": context.model,
        "task_subset": context.task_subset,
        "task_id": "",
        "repeat": context.repeat,
        "passed": None,
        "runtime_seconds": command_result.runtime_seconds,
        "stdout_log": str(command_result.stdout_log),
        "stderr_log": str(command_result.stderr_log),
        "task_log_dir": "",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
        "timestamp": utc_now(),
        "git_commit": context.git_commit,
        "terminal_bench_commit": tb_commit,
        "exit_code": command_result.exit_code,
        "output_dir": str(context.run_dir),
        "command": shlex.join(context.command),
        "error": error,
    }


def fallback_rows(
    context: RunContext,
    command_result: CommandResult,
    task_ids: List[str],
    dry_run: bool,
) -> List[Dict[str, Any]]:
    if dry_run:
        error = "dry_run"
        exit_code = 0
    else:
        error = "no Terminal-Bench result rows found"
        exit_code = command_result.exit_code

    fake_result = CommandResult(
        exit_code=exit_code,
        runtime_seconds=command_result.runtime_seconds,
        stdout_log=command_result.stdout_log,
        stderr_log=command_result.stderr_log,
    )
    tb_commit = terminal_bench_commit(context.run_dir)
    row_task_ids = task_ids or ["__all__"]
    rows = []
    for task_id in row_task_ids:
        row = base_row(context, fake_result, tb_commit, error=error)
        row["task_id"] = task_id
        row["passed"] = None if dry_run else False
        rows.append(row)
    return rows


def normalize_for_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def append_results(jsonl_path: Path, csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_for_csv(row.get(key)) for key in RESULT_COLUMNS})


def make_context(
    args: argparse.Namespace,
    run_id: str,
    harness_name: str,
    harness_version: str,
    model: str,
    task_subset: str,
    repeat: int,
    command: List[str],
    repo_commit: Optional[str],
) -> RunContext:
    tb_run_id = command[command.index("--run-id") + 1]
    combo = "__".join(
        [
            safe_name(harness_name),
            safe_name(harness_version),
            safe_name(model),
            safe_name(task_subset),
            "repeat-%d" % repeat,
            tb_run_id,
        ]
    )
    log_dir = args.output_dir / "logs" / combo
    return RunContext(
        run_id=run_id,
        tb_run_id=tb_run_id,
        harness_name=harness_name,
        harness_version=harness_version,
        model=model,
        task_subset=task_subset,
        repeat=repeat,
        output_root=args.output_dir,
        run_dir=args.tbench_runs_dir / tb_run_id,
        stdout_log=log_dir / "stdout.log",
        stderr_log=log_dir / "stderr.log",
        command=command,
        git_commit=repo_commit,
    )


def run_one(
    args: argparse.Namespace,
    context: RunContext,
    pricing: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    if args.dry_run:
        context.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        context.stdout_log.write_text(shlex.join(context.command) + "\n", encoding="utf-8")
        context.stderr_log.write_text("", encoding="utf-8")
        command_result = CommandResult(0, 0.0, context.stdout_log, context.stderr_log)
        return fallback_rows(context, command_result, split_csv(args.task_ids), dry_run=True)

    command_result = run_command(context.command, context.stdout_log, context.stderr_log)
    rows = parse_aggregate_results(context, command_result, pricing)
    if not rows:
        rows = parse_trial_result_files(context, command_result, pricing)
    if not rows:
        rows = fallback_rows(context, command_result, split_csv(args.task_ids), dry_run=False)
    return rows


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench task matrices across harness versions and models."
    )
    parser.add_argument("--harness", required=True, help="Harness name, e.g. codex-cli")
    parser.add_argument(
        "--harness-versions",
        required=True,
        help="Comma-separated harness versions or refs, e.g. 0.135.0,latest",
    )
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated Terminal-Bench datasets/subsets, e.g. terminal-bench-core",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--dataset-version",
        default="head",
        help="Dataset version to append when --tasks omits ==version",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated task IDs or globs passed as repeated --task-id flags",
    )
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Optional local Terminal-Bench dataset path; overrides --dataset",
    )
    parser.add_argument("--tb-command", default="tb", help="Terminal-Bench CLI command")
    parser.add_argument(
        "--tb-extra-args",
        default="",
        help="Extra arguments appended to 'tb run', e.g. '--log-level debug'",
    )
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--tbench-runs-dir",
        type=Path,
        default=Path("outputs/tbench_runs"),
        help="Directory passed to Terminal-Bench --output-path",
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=Path("outputs/results.jsonl"),
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("outputs/results.csv"),
    )
    parser.add_argument(
        "--pricing-file",
        default="",
        help="Optional JSON pricing file with per-million token rates",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned commands and result rows without running Terminal-Bench",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tbench_runs_dir.mkdir(parents=True, exist_ok=True)
    pricing = load_pricing(args.pricing_file)
    repo_commit = git_commit(Path.cwd())
    run_id = str(uuid.uuid4())

    total_rows = 0
    started = compact_utc_now()

    for harness_name in split_csv(args.harness):
        harness = get_harness(harness_name)
        for harness_version in split_csv(args.harness_versions):
            for model in split_csv(args.models):
                for task_subset in split_csv(args.tasks):
                    for repeat in range(1, args.repeats + 1):
                        tb_run_id = "__".join(
                            [
                                started,
                                safe_name(harness_name),
                                safe_name(harness_version),
                                safe_name(model),
                                safe_name(task_subset),
                                "r%d" % repeat,
                            ]
                        )
                        command = build_tb_command(
                            args,
                            harness,
                            harness_version,
                            model,
                            task_subset,
                            tb_run_id,
                        )
                        context = make_context(
                            args,
                            run_id,
                            harness_name,
                            harness_version,
                            model,
                            task_subset,
                            repeat,
                            command,
                            repo_commit,
                        )
                        print("Running: %s" % shlex.join(command), flush=True)
                        rows = run_one(args, context, pricing)
                        append_results(args.results_jsonl, args.results_csv, rows)
                        total_rows += len(rows)

    print(
        "Wrote %d result rows to %s and %s"
        % (total_rows, args.results_jsonl, args.results_csv),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
