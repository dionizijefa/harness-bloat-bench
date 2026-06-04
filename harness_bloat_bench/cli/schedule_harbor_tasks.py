#!/usr/bin/env python3
"""
Lightweight resource-aware scheduler for Terminal-Bench 2 / Harbor tasks.

The scheduler discovers local task.toml files, optionally downloads a Harbor
dataset first, and runs each admitted task as its own `harbor run` process with
`--n-concurrent 1`. CPU, memory, storage, and GPU requirements come from the
task TOML. Host CPU/RAM/storage/GPU capacity is tracked in this process.
"""

import argparse
import csv
import datetime as dt
import fnmatch
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
]

RESULT_COLUMNS = [
    "run_id",
    "experiment_id",
    "scheduler_run_id",
    "task_id",
    "task_name",
    "task_version_id",
    "task_domain",
    "task_path",
    "attempt",
    "status",
    "passed",
    "completed",
    "exit_code",
    "runtime_seconds",
    "cpus",
    "memory_mb",
    "storage_mb",
    "gpus",
    "agent_timeout_sec",
    "verifier_timeout_sec",
    "environment_build_timeout_sec",
    "verifier_environment_timeout_sec",
    "job_name",
    "job_dir",
    "stdout_log",
    "stderr_log",
    "result_json",
    "task_log_dir",
    "model_version",
    "harness_version",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "command",
    "error",
    "started_at",
    "finished_at",
    "completed_at",
    "harness_events",
]


@dataclass(frozen=True)
class Resources:
    cpus: int = 0
    memory_mb: int = 0
    storage_mb: int = 0
    gpus: int = 0

    def fits_within(self, other: "Resources") -> bool:
        return (
            self.cpus <= other.cpus
            and self.memory_mb <= other.memory_mb
            and self.storage_mb <= other.storage_mb
            and self.gpus <= other.gpus
        )

    def add(self, other: "Resources") -> "Resources":
        return Resources(
            cpus=self.cpus + other.cpus,
            memory_mb=self.memory_mb + other.memory_mb,
            storage_mb=self.storage_mb + other.storage_mb,
            gpus=self.gpus + other.gpus,
        )

    def subtract(self, other: "Resources") -> "Resources":
        return Resources(
            cpus=self.cpus - other.cpus,
            memory_mb=self.memory_mb - other.memory_mb,
            storage_mb=self.storage_mb - other.storage_mb,
            gpus=self.gpus - other.gpus,
        )


@dataclass
class TaskSpec:
    name: str
    path: Path
    config_path: Path
    resources: Resources
    agent_timeout_sec: Optional[float]
    verifier_timeout_sec: Optional[float]
    environment_build_timeout_sec: Optional[float]
    verifier_environment_timeout_sec: Optional[float]
    raw: dict[str, Any] = field(repr=False)


@dataclass
class RunningTask:
    task: TaskSpec
    attempt: int
    resources: Resources
    process: subprocess.Popen[None]
    command: list[str]
    stdout_log: Path
    stderr_log: Path
    job_name: str
    job_dir: Path
    started_at: str
    start_time: float


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compact_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._-") or "value"


def split_patterns(values: Iterable[str]) -> list[str]:
    patterns: list[str] = []
    for value in values:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return patterns


def parse_size_mb(value: str | int | float | None, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(math.ceil(float(value)))

    raw = str(value).strip()
    if not raw:
        return default

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)", raw)
    if not match:
        raise ValueError(
            "invalid size %r; use values like 4096MB, 128GB, or 1TB" % value
        )

    number = float(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "": 1,
        "m": 1,
        "mb": 1,
        "mi": 1,
        "mib": 1,
        "g": 1024,
        "gb": 1024,
        "gi": 1024,
        "gib": 1024,
        "t": 1024 * 1024,
        "tb": 1024 * 1024,
        "ti": 1024 * 1024,
        "tib": 1024 * 1024,
    }
    if unit not in multipliers:
        raise ValueError("unknown size unit %r in %r" % (unit, value))
    return int(math.ceil(number * multipliers[unit]))


def as_int(value: Any, *, default: int = 0, minimum: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return max(minimum, int(math.ceil(float(value))))
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def parse_task(path: Path) -> TaskSpec:
    config_path = path / "task.toml" if path.is_dir() else path
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    task_section = section(data, "task")
    environment = section(data, "environment")
    verifier = section(data, "verifier")
    verifier_environment = section(verifier, "environment")

    task_name = str(task_section.get("name") or config_path.parent.name)
    cpus = max(
        1,
        as_int(environment.get("cpus"), default=1, minimum=1),
        as_int(verifier_environment.get("cpus"), default=0, minimum=0),
    )
    memory_mb = max(
        1,
        as_int(environment.get("memory_mb"), default=1024, minimum=1),
        as_int(verifier_environment.get("memory_mb"), default=0, minimum=0),
    )
    storage_mb = max(
        0,
        as_int(environment.get("storage_mb"), default=0, minimum=0),
        as_int(verifier_environment.get("storage_mb"), default=0, minimum=0),
    )
    gpus = max(
        0,
        as_int(environment.get("gpus"), default=0, minimum=0),
        as_int(verifier_environment.get("gpus"), default=0, minimum=0),
    )

    return TaskSpec(
        name=task_name,
        path=config_path.parent,
        config_path=config_path,
        resources=Resources(
            cpus=cpus, memory_mb=memory_mb, storage_mb=storage_mb, gpus=gpus
        ),
        agent_timeout_sec=as_float(section(data, "agent").get("timeout_sec")),
        verifier_timeout_sec=as_float(verifier.get("timeout_sec")),
        environment_build_timeout_sec=as_float(environment.get("build_timeout_sec")),
        verifier_environment_timeout_sec=as_float(
            verifier_environment.get("timeout_sec")
            or verifier_environment.get("build_timeout_sec")
        ),
        raw=data,
    )


def discover_tasks(root: Path) -> list[TaskSpec]:
    task_paths = sorted(root.rglob("task.toml")) if root.is_dir() else [root]
    tasks: list[TaskSpec] = []
    seen: set[Path] = set()
    for config_path in task_paths:
        if config_path in seen:
            continue
        seen.add(config_path)
        try:
            tasks.append(parse_task(config_path))
        except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
            print("Skipping %s: %s" % (config_path, exc), file=sys.stderr, flush=True)
    return tasks


def task_matches(task: TaskSpec, include: list[str], exclude: list[str]) -> bool:
    names = [task.name, task.name.split("/")[-1], str(task.path)]
    included = (
        True
        if not include
        else any(
            fnmatch.fnmatchcase(name, pattern) for pattern in include for name in names
        )
    )
    excluded = any(
        fnmatch.fnmatchcase(name, pattern) for pattern in exclude for name in names
    )
    return included and not excluded


def ensure_dataset(args: argparse.Namespace) -> Path:
    if args.dataset_path:
        path = args.dataset_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("--dataset-path does not exist: %s" % path)
        return path

    dataset_dir = args.download_dir.expanduser().resolve() / safe_name(args.dataset)
    existing_tasks = (
        list(dataset_dir.rglob("task.toml")) if dataset_dir.exists() else []
    )
    if existing_tasks and not args.overwrite_dataset:
        return dataset_dir

    if not args.fetch:
        raise FileNotFoundError(
            "no local tasks found under %s and --no-fetch was set" % dataset_dir
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.split(args.harbor_command) + [
        "download",
        args.dataset,
        "--output-dir",
        str(dataset_dir),
    ]
    if args.overwrite_dataset:
        command.append("--overwrite")
    if args.registry_url:
        command.extend(["--registry-url", args.registry_url])
    if args.registry_path:
        command.extend(["--registry-path", str(args.registry_path)])

    print("Fetching dataset: %s" % shlex.join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "dataset download failed with exit code %d" % completed.returncode
        )
    return dataset_dir


def thread_env(cpus: int) -> dict[str, str]:
    threads = str(max(1, int(cpus)))
    env = {key: threads for key in THREAD_ENV_KEYS}
    env["MAKEFLAGS"] = "-j%s" % threads
    return env


def write_harbor_config(run_config_dir: Path, task: TaskSpec) -> Path:
    run_config_dir.mkdir(parents=True, exist_ok=True)
    env = thread_env(task.resources.cpus)
    config = {
        "environment": {"env": env},
        "verifier": {"env": env},
    }
    path = run_config_dir / ("%s.json" % safe_name(task.name))
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def build_harbor_command(
    args: argparse.Namespace,
    task: TaskSpec,
    job_name: str,
    config_path: Path,
) -> list[str]:
    command = shlex.split(args.harbor_command) + [
        "run",
        "--path",
        str(task.path),
        "--config",
        str(config_path),
        "--jobs-dir",
        str(args.jobs_dir),
        "--job-name",
        job_name,
        "--n-attempts",
        str(args.n_attempts),
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--cpus",
        args.cpu_mode,
        "--memory",
        args.memory_mode,
        "--override-cpus",
        str(task.resources.cpus),
        "--override-memory",
        str(task.resources.memory_mb),
        "--override-storage",
        str(task.resources.storage_mb),
        "--override-gpus",
        str(task.resources.gpus),
    ]

    agent = args.agent
    agent_import_path = args.agent_import_path
    if args.openrouter and agent == "codex" and not agent_import_path:
        agent = ""
        agent_import_path = "harness_bloat_bench.openrouter_codex:OpenRouterCodex"

    if agent:
        command.extend(["--agent", agent])
    if agent_import_path:
        command.extend(["--agent-import-path", agent_import_path])
    if args.model:
        command.extend(["--model", args.model])
    for value in args.agent_kwarg:
        command.extend(["--agent-kwarg", value])
    for value in args.agent_env:
        command.extend(["--agent-env", value])
    for value in args.verifier_env:
        command.extend(["--verifier-env", value])
    if args.yes:
        command.append("--yes")
    if args.quiet:
        command.append("--quiet")
    if args.harbor_extra_args:
        command.extend(shlex.split(args.harbor_extra_args))
    return command


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    cwd = str(Path.cwd())
    env["PYTHONPATH"] = cwd + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    if args.openrouter:
        key = env.get(args.openrouter_api_key_env)
        if key:
            env["OPENAI_API_KEY"] = key
        elif not args.dry_run:
            raise RuntimeError(
                "%s is required for --openrouter runs" % args.openrouter_api_key_env
            )
        env["OPENAI_BASE_URL"] = args.openrouter_base_url
    return env


def load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def first_present(data: dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def first_nested_present(data: Any, paths: Iterable[Sequence[str]]) -> Optional[Any]:
    for path in paths:
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def as_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def harbor_passed(data: Any) -> Optional[bool]:
    if not isinstance(data, dict):
        return None
    verifier_result = data.get("verifier_result")
    if isinstance(verifier_result, dict):
        rewards = verifier_result.get("rewards")
        if isinstance(rewards, dict):
            value = rewards.get("reward")
            if value is None and rewards:
                value = min(rewards.values())
            try:
                return float(value) >= 1.0
            except (TypeError, ValueError):
                return None
    if "is_resolved" in data:
        return bool(data["is_resolved"])
    return None


def agent_contexts(data: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = []
    if isinstance(data.get("agent_result"), dict):
        contexts.append(data["agent_result"])
    step_results = data.get("step_results")
    if isinstance(step_results, list):
        for step_result in step_results:
            if not isinstance(step_result, dict):
                continue
            agent_result = step_result.get("agent_result")
            if isinstance(agent_result, dict):
                contexts.append(agent_result)
    return contexts


def reasoning_tokens_from_context(context: dict[str, Any]) -> Optional[int]:
    return as_optional_int(
        first_present(
            context,
            [
                "n_reasoning_tokens",
                "reasoning_tokens",
                "total_reasoning_tokens",
            ],
        )
        or first_nested_present(
            context,
            [
                ("usage", "reasoning_tokens"),
                ("usage", "completion_tokens_details", "reasoning_tokens"),
                ("response", "usage", "reasoning_tokens"),
                (
                    "response",
                    "usage",
                    "completion_tokens_details",
                    "reasoning_tokens",
                ),
            ],
        )
    )


def token_details(data: dict[str, Any]) -> dict[str, Optional[int]]:
    contexts = agent_contexts(data)
    if contexts:
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        reasoning_tokens: Optional[int] = None
        for context in contexts:
            current_input = as_optional_int(context.get("n_input_tokens"))
            current_output = as_optional_int(context.get("n_output_tokens"))
            current_reasoning = reasoning_tokens_from_context(context)
            if current_input is not None:
                input_tokens = (input_tokens or 0) + current_input
            if current_output is not None:
                output_tokens = (output_tokens or 0) + current_output
            if current_reasoning is not None:
                reasoning_tokens = (reasoning_tokens or 0) + current_reasoning
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    return {
        "input_tokens": as_optional_int(
            first_present(
                data, ["total_input_tokens", "input_tokens", "n_input_tokens"]
            )
        ),
        "output_tokens": as_optional_int(
            first_present(
                data, ["total_output_tokens", "output_tokens", "n_output_tokens"]
            )
        ),
        "reasoning_tokens": as_optional_int(
            first_present(
                data,
                [
                    "total_reasoning_tokens",
                    "reasoning_tokens",
                    "n_reasoning_tokens",
                ],
            )
            or first_nested_present(
                data,
                [
                    ("usage", "reasoning_tokens"),
                    ("usage", "completion_tokens_details", "reasoning_tokens"),
                ],
            )
        ),
    }


def harness_events_from_result(data: dict[str, Any]) -> Optional[Any]:
    explicit = first_present(
        data,
        ["harness_events", "agent_events", "events", "codex_events"],
    )
    if explicit is not None:
        return explicit

    contexts = agent_contexts(data)
    context_events = []
    for context in contexts:
        event_value = first_present(
            context,
            ["harness_events", "agent_events", "events", "codex_events", "messages"],
        )
        if event_value is not None:
            context_events.append(event_value)
    if context_events:
        return context_events[0] if len(context_events) == 1 else context_events
    if contexts:
        return {"agent_results": contexts}
    return None


def empty_result_details(job_dir: Path) -> dict[str, Any]:
    result_json = job_dir / "result.json"
    return {
        "passed": None,
        "result_json": str(result_json) if result_json.exists() else "",
        "task_log_dir": "",
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "started_at": None,
        "completed_at": None,
        "harness_events": None,
    }


def details_from_data(
    data: dict[str, Any], result_json: Path, task_log_dir: str
) -> dict[str, Any]:
    return {
        "passed": harbor_passed(data),
        "result_json": str(result_json),
        "task_log_dir": task_log_dir,
        **token_details(data),
        "started_at": first_present(
            data, ["started_at", "trial_started_at", "agent_started_at"]
        ),
        "completed_at": first_present(
            data,
            [
                "finished_at",
                "completed_at",
                "trial_ended_at",
                "agent_ended_at",
            ],
        ),
        "harness_events": harness_events_from_result(data),
    }


def result_details(job_dir: Path) -> dict[str, Any]:
    result_json = job_dir / "result.json"
    data = load_json(result_json)
    if isinstance(data, dict):
        trial_results = data.get("trial_results")
        if isinstance(trial_results, list) and trial_results:
            first = next(
                (item for item in trial_results if isinstance(item, dict)), None
            )
            if first:
                trial_name = first.get("trial_name")
                task_log_dir = ""
                if trial_name and (job_dir / str(trial_name)).exists():
                    task_log_dir = str(job_dir / str(trial_name))
                return details_from_data(first, result_json, task_log_dir)

    aggregate = load_json(job_dir / "results.json")
    if isinstance(aggregate, dict):
        results = aggregate.get("results")
        if isinstance(results, list) and results:
            first = next((item for item in results if isinstance(item, dict)), None)
            if first:
                task_id = first.get("task_id")
                trial_name = first.get("trial_name")
                log_dir = ""
                if (
                    task_id
                    and trial_name
                    and (job_dir / str(task_id) / str(trial_name)).exists()
                ):
                    log_dir = str(job_dir / str(task_id) / str(trial_name))
                elif task_id and (job_dir / str(task_id)).exists():
                    log_dir = str(job_dir / str(task_id))
                return details_from_data(first, job_dir / "results.json", log_dir)

    for path in sorted(job_dir.glob("*/result.json")):
        data = load_json(path)
        if isinstance(data, dict):
            return details_from_data(data, path, str(path.parent))

    return empty_result_details(job_dir)


def task_version_id(task: TaskSpec) -> str:
    task_section = section(task.raw, "task")
    value = first_present(
        task_section,
        ["version_id", "version", "task_version_id", "task_version"],
    )
    if value:
        return str(value)
    return str(task.config_path)


def task_domain(task: TaskSpec, dataset: str) -> str:
    if "/" in task.name:
        return task.name.split("/", 1)[0]
    if dataset:
        return dataset.split("/", 1)[0]
    return task.path.parent.name


def harness_version(agent_kwargs: Iterable[str]) -> str:
    version = ""
    for value in agent_kwargs:
        if "=" not in value:
            continue
        key, raw = value.split("=", 1)
        if key.strip() == "version" and raw.strip():
            version = raw.strip()
    return version


def result_row(
    args: argparse.Namespace,
    scheduler_run_id: str,
    task: TaskSpec,
    attempt: int,
    status: str,
    exit_code: Optional[int],
    runtime_seconds: float,
    stdout_log: Path,
    stderr_log: Path,
    job_name: str,
    job_dir: Path,
    command: list[str],
    error: str,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    if job_name and job_dir.exists():
        details = result_details(job_dir)
    else:
        details = empty_result_details(job_dir)
    completed = status not in {"dry_run", "skipped"} and exit_code == 0
    return {
        "run_id": job_name or scheduler_run_id,
        "experiment_id": args.experiment_id or scheduler_run_id,
        "scheduler_run_id": scheduler_run_id,
        "task_id": task.name,
        "task_name": task.name,
        "task_version_id": task_version_id(task),
        "task_domain": task_domain(task, args.dataset),
        "task_path": str(task.path),
        "attempt": attempt,
        "status": status,
        "passed": details["passed"],
        "completed": completed,
        "exit_code": exit_code,
        "runtime_seconds": runtime_seconds,
        "cpus": task.resources.cpus,
        "memory_mb": task.resources.memory_mb,
        "storage_mb": task.resources.storage_mb,
        "gpus": task.resources.gpus,
        "agent_timeout_sec": task.agent_timeout_sec,
        "verifier_timeout_sec": task.verifier_timeout_sec,
        "environment_build_timeout_sec": task.environment_build_timeout_sec,
        "verifier_environment_timeout_sec": task.verifier_environment_timeout_sec,
        "job_name": job_name,
        "job_dir": str(job_dir),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "result_json": details["result_json"],
        "task_log_dir": details["task_log_dir"],
        "model_version": args.model,
        "harness_version": harness_version(args.agent_kwarg),
        "input_tokens": details["input_tokens"],
        "output_tokens": details["output_tokens"],
        "reasoning_tokens": details["reasoning_tokens"],
        "command": shlex.join(command),
        "error": error,
        "started_at": details["started_at"] or started_at,
        "finished_at": finished_at,
        "completed_at": details["completed_at"] or finished_at,
        "harness_events": details["harness_events"],
    }


def normalize_for_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def append_results(json_path: Path, csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_json(json_path)
    all_rows = existing if isinstance(existing, list) else []
    all_rows.extend(rows)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(all_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(json_path)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: normalize_for_csv(row.get(key)) for key in RESULT_COLUMNS}
            )


def resources_dict(resources: Resources) -> dict[str, int]:
    return {
        "cpus": resources.cpus,
        "memory_mb": resources.memory_mb,
        "storage_mb": resources.storage_mb,
        "gpus": resources.gpus,
    }


def task_resource_dict(task: TaskSpec) -> dict[str, Any]:
    return {
        **resources_dict(task.resources),
        "agent_timeout_sec": task.agent_timeout_sec,
        "verifier_timeout_sec": task.verifier_timeout_sec,
        "environment_build_timeout_sec": task.environment_build_timeout_sec,
        "verifier_environment_timeout_sec": task.verifier_environment_timeout_sec,
    }


def pending_task_snapshot(task: TaskSpec, attempt: int) -> dict[str, Any]:
    return {
        "task_name": task.name,
        "task_path": str(task.path),
        "attempt": attempt,
        "status": "pending",
        "runtime_seconds": 0.0,
        "resources": task_resource_dict(task),
    }


def running_task_snapshot(item: RunningTask) -> dict[str, Any]:
    return {
        "task_name": item.task.name,
        "task_path": str(item.task.path),
        "attempt": item.attempt,
        "status": "running",
        "runtime_seconds": time.perf_counter() - item.start_time,
        "resources": task_resource_dict(item.task),
        "job_name": item.job_name,
        "job_dir": str(item.job_dir),
        "stdout_log": str(item.stdout_log),
        "stderr_log": str(item.stderr_log),
        "command": shlex.join(item.command),
        "started_at": item.started_at,
    }


def row_task_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "experiment_id": row.get("experiment_id"),
        "scheduler_run_id": row.get("scheduler_run_id"),
        "task_id": row.get("task_id"),
        "task_name": row.get("task_name"),
        "task_version_id": row.get("task_version_id"),
        "task_domain": row.get("task_domain"),
        "task_path": row.get("task_path"),
        "attempt": row.get("attempt"),
        "status": row.get("status"),
        "passed": row.get("passed"),
        "completed": row.get("completed"),
        "exit_code": row.get("exit_code"),
        "runtime_seconds": row.get("runtime_seconds") or 0.0,
        "resources": {
            "cpus": row.get("cpus") or 0,
            "memory_mb": row.get("memory_mb") or 0,
            "storage_mb": row.get("storage_mb") or 0,
            "gpus": row.get("gpus") or 0,
            "agent_timeout_sec": row.get("agent_timeout_sec"),
            "verifier_timeout_sec": row.get("verifier_timeout_sec"),
            "environment_build_timeout_sec": row.get("environment_build_timeout_sec"),
            "verifier_environment_timeout_sec": row.get(
                "verifier_environment_timeout_sec"
            ),
        },
        "job_name": row.get("job_name"),
        "job_dir": row.get("job_dir"),
        "stdout_log": row.get("stdout_log"),
        "stderr_log": row.get("stderr_log"),
        "result_json": row.get("result_json"),
        "task_log_dir": row.get("task_log_dir"),
        "model_version": row.get("model_version"),
        "harness_version": row.get("harness_version"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "reasoning_tokens": row.get("reasoning_tokens"),
        "command": row.get("command"),
        "error": row.get("error"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "completed_at": row.get("completed_at"),
        "harness_events": row.get("harness_events"),
    }


def state_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [str(task.get("status") or "unknown") for task in tasks]
    return {
        "total": len(tasks),
        "pending": statuses.count("pending"),
        "running": statuses.count("running"),
        "passed": statuses.count("passed"),
        "failed": statuses.count("failed"),
        "skipped": statuses.count("skipped"),
        "dry_run": statuses.count("dry_run"),
        "finished": sum(
            1
            for status in statuses
            if status in {"passed", "failed", "skipped", "dry_run"}
        ),
    }


def write_scheduler_state(
    args: argparse.Namespace,
    scheduler_run_id: str,
    started_at: str,
    status: str,
    capacity: Resources,
    used: Resources,
    pending: list[tuple[TaskSpec, int]],
    running: list[RunningTask],
    completed_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    *,
    message: str = "",
) -> None:
    if not args.state_json:
        return

    tasks = (
        [row_task_snapshot(row) for row in skipped_rows]
        + [row_task_snapshot(row) for row in completed_rows]
        + [running_task_snapshot(item) for item in running]
        + [pending_task_snapshot(task, attempt) for task, attempt in pending]
    )
    state = {
        "scheduler_run_id": scheduler_run_id,
        "status": status,
        "message": message,
        "started_at": started_at,
        "updated_at": utc_now(),
        "capacity": resources_dict(capacity),
        "used": resources_dict(used),
        "available": resources_dict(capacity.subtract(used)),
        "counts": state_counts(tasks),
        "results_json": str(args.results_json),
        "results_csv": str(args.results_csv),
        "jobs_dir": str(args.jobs_dir),
        "logs_dir": str(args.logs_dir),
        "tasks": tasks,
    }

    args.state_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.state_json.with_suffix(args.state_json.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.state_json)


def start_task(
    args: argparse.Namespace,
    started_stamp: str,
    task: TaskSpec,
    attempt: int,
    env: dict[str, str],
) -> RunningTask:
    job_name = "__".join([started_stamp, safe_name(task.name), "a%d" % attempt])
    config_path = write_harbor_config(args.scheduler_config_dir, task)
    command = build_harbor_command(args, task, job_name, config_path)
    log_dir = args.logs_dir / ("%s__a%d" % (safe_name(task.name), attempt))
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    args.jobs_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Starting %s attempt %d with %d CPU, %d MB RAM, %d MB storage"
        % (
            task.name,
            attempt,
            task.resources.cpus,
            task.resources.memory_mb,
            task.resources.storage_mb,
        ),
        flush=True,
    )
    with (
        stdout_log.open("w", encoding="utf-8") as out,
        stderr_log.open("w", encoding="utf-8") as err,
    ):
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env)

    return RunningTask(
        task=task,
        attempt=attempt,
        resources=task.resources,
        process=process,
        command=command,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        job_name=job_name,
        job_dir=args.jobs_dir / job_name,
        started_at=utc_now(),
        start_time=time.perf_counter(),
    )


def dry_run_rows(
    args: argparse.Namespace,
    scheduler_run_id: str,
    started_stamp: str,
    tasks: list[TaskSpec],
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        job_name = "__".join([started_stamp, safe_name(task.name), "a1"])
        config_path = write_harbor_config(args.scheduler_config_dir, task)
        command = build_harbor_command(args, task, job_name, config_path)
        log_dir = args.logs_dir / ("%s__a1" % safe_name(task.name))
        stdout_log = log_dir / "stdout.log"
        stderr_log = log_dir / "stderr.log"
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text(shlex.join(command) + "\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        rows.append(
            result_row(
                args,
                scheduler_run_id,
                task,
                1,
                "dry_run",
                0,
                0.0,
                stdout_log,
                stderr_log,
                job_name,
                args.jobs_dir / job_name,
                command,
                "dry_run",
                utc_now(),
                utc_now(),
            )
        )
    return rows


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resource-aware scheduler for Terminal-Bench 2 / Harbor tasks."
    )
    parser.add_argument(
        "--dataset",
        default="terminal-bench/terminal-bench-2-1",
        help="Harbor dataset to fetch when --dataset-path is not supplied.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Local dataset or task directory containing task.toml files.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("outputs/scheduler/downloads"),
        help="Directory used for Harbor dataset downloads.",
    )
    parser.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--registry-url", default="")
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Task include glob. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Task exclude glob. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=0, help="Maximum selected tasks to queue."
    )
    parser.add_argument("--total-cpus", type=int, default=64)
    parser.add_argument("--total-memory", default="128GB")
    parser.add_argument("--total-storage", default="1TB")
    parser.add_argument("--total-gpus", type=int, default=0)
    parser.add_argument("--reserve-cpus", type=int, default=0)
    parser.add_argument("--reserve-memory", default="0GB")
    parser.add_argument("--reserve-gpus", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--harbor-command", default="harbor")
    parser.add_argument("--harbor-extra-args", default="")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--agent-import-path", default="")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=["version=latest"],
        help="Harbor --agent-kwarg value. Repeat for multiple values.",
    )
    parser.add_argument("--agent-env", action="append", default=[])
    parser.add_argument("--verifier-env", action="append", default=[])
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument(
        "--cpu-mode",
        default="limit",
        choices=["auto", "limit", "request", "guarantee", "ignore"],
    )
    parser.add_argument(
        "--memory-mode",
        default="limit",
        choices=["auto", "limit", "request", "guarantee", "ignore"],
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--openrouter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map OPENROUTER_API_KEY to OPENAI_API_KEY and set OPENAI_BASE_URL.",
    )
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scheduler"))
    parser.add_argument("--jobs-dir", type=Path, default=Path("outputs/scheduler/jobs"))
    parser.add_argument("--logs-dir", type=Path, default=Path("outputs/scheduler/logs"))
    parser.add_argument(
        "--scheduler-config-dir",
        type=Path,
        default=Path("outputs/scheduler/configs"),
        help="Directory for generated Harbor JSON configs.",
    )
    parser.add_argument(
        "--results-json", type=Path, default=Path("outputs/scheduler/results.json")
    )
    parser.add_argument(
        "--results-csv", type=Path, default=Path("outputs/scheduler/results.csv")
    )
    parser.add_argument(
        "--experiment-id",
        default="",
        help=(
            "Optional stable experiment identifier to write on every result row. "
            "Defaults to this scheduler run id."
        ),
    )
    parser.add_argument(
        "--state-json", type=Path, default=Path("outputs/scheduler/state.json")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")

    total = Resources(
        cpus=args.total_cpus,
        memory_mb=parse_size_mb(args.total_memory),
        storage_mb=parse_size_mb(args.total_storage),
        gpus=args.total_gpus,
    )
    reserve = Resources(
        cpus=args.reserve_cpus,
        memory_mb=parse_size_mb(args.reserve_memory),
        storage_mb=0,
        gpus=args.reserve_gpus,
    )
    capacity = total.subtract(reserve)
    if min(capacity.cpus, capacity.memory_mb, capacity.storage_mb, capacity.gpus) < 0:
        raise ValueError("reserve resources exceed total resources")

    dataset_root = ensure_dataset(args)
    include = split_patterns(args.include)
    exclude = split_patterns(args.exclude)
    tasks = [
        task
        for task in discover_tasks(dataset_root)
        if task_matches(task, include, exclude)
    ]
    if args.n_tasks > 0:
        tasks = tasks[: args.n_tasks]
    if not tasks:
        raise RuntimeError("no tasks selected under %s" % dataset_root)

    runnable: list[TaskSpec] = []
    skipped_rows: list[dict[str, Any]] = []
    started_stamp = compact_utc_now()
    started_at = utc_now()
    scheduler_run_id = str(uuid.uuid4())
    for task in tasks:
        if task.resources.fits_within(capacity):
            runnable.append(task)
        else:
            now = utc_now()
            skipped_rows.append(
                result_row(
                    args,
                    scheduler_run_id,
                    task,
                    0,
                    "skipped",
                    None,
                    0.0,
                    args.logs_dir / ("%s__skipped_stdout.log" % safe_name(task.name)),
                    args.logs_dir / ("%s__skipped_stderr.log" % safe_name(task.name)),
                    "",
                    args.jobs_dir,
                    [],
                    "resource requirements exceed scheduler capacity",
                    now,
                    now,
                )
            )

    print(
        "Selected %d tasks (%d runnable, %d skipped). Capacity: %d CPU, %d MB RAM, %d MB storage, %d GPU."
        % (
            len(tasks),
            len(runnable),
            len(skipped_rows),
            capacity.cpus,
            capacity.memory_mb,
            capacity.storage_mb,
            capacity.gpus,
        ),
        flush=True,
    )

    if args.dry_run:
        rows = skipped_rows + dry_run_rows(
            args, scheduler_run_id, started_stamp, runnable
        )
        append_results(args.results_json, args.results_csv, rows)
        write_scheduler_state(
            args,
            scheduler_run_id,
            started_at,
            "complete",
            capacity,
            Resources(),
            [],
            [],
            rows,
            [],
            message="dry_run",
        )
        print(
            "Dry run wrote %d rows to %s and %s"
            % (len(rows), args.results_json, args.results_csv),
            flush=True,
        )
        return 0

    env = command_env(args)
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    append_results(args.results_json, args.results_csv, skipped_rows)

    pending: list[tuple[TaskSpec, int]] = [(task, 1) for task in runnable]
    running: list[RunningTask] = []
    used = Resources()
    completed_rows: list[dict[str, Any]] = []
    write_scheduler_state(
        args,
        scheduler_run_id,
        started_at,
        "running",
        capacity,
        used,
        pending,
        running,
        completed_rows,
        skipped_rows,
        message="selected %d runnable tasks" % len(runnable),
    )

    while pending or running:
        launched = True
        while launched:
            launched = False
            available = capacity.subtract(used)
            for index, (task, attempt) in enumerate(pending):
                if task.resources.fits_within(available):
                    pending.pop(index)
                    running_task = start_task(args, started_stamp, task, attempt, env)
                    running.append(running_task)
                    used = used.add(task.resources)
                    launched = True
                    write_scheduler_state(
                        args,
                        scheduler_run_id,
                        started_at,
                        "running",
                        capacity,
                        used,
                        pending,
                        running,
                        completed_rows,
                        skipped_rows,
                        message="started %s attempt %d" % (task.name, attempt),
                    )
                    break

        if not running:
            break

        time.sleep(1.0)
        still_running: list[RunningTask] = []
        finished_rows: list[dict[str, Any]] = []
        for item in running:
            exit_code = item.process.poll()
            if exit_code is None:
                still_running.append(item)
                continue

            used = used.subtract(item.resources)
            runtime = time.perf_counter() - item.start_time
            finished_at = utc_now()
            status = "passed" if exit_code == 0 else "failed"
            error = "" if exit_code == 0 else "harbor exited with code %d" % exit_code
            row = result_row(
                args,
                scheduler_run_id,
                item.task,
                item.attempt,
                status,
                exit_code,
                runtime,
                item.stdout_log,
                item.stderr_log,
                item.job_name,
                item.job_dir,
                item.command,
                error,
                item.started_at,
                finished_at,
            )
            finished_rows.append(row)
            print(
                "Finished %s attempt %d: %s (exit %d)"
                % (item.task.name, item.attempt, status, exit_code),
                flush=True,
            )
            if exit_code != 0 and item.attempt <= args.max_retries:
                pending.append((item.task, item.attempt + 1))

        if finished_rows:
            append_results(args.results_json, args.results_csv, finished_rows)
            completed_rows.extend(finished_rows)
        running = still_running
        write_scheduler_state(
            args,
            scheduler_run_id,
            started_at,
            "running",
            capacity,
            used,
            pending,
            running,
            completed_rows,
            skipped_rows,
        )

    failures = sum(1 for row in completed_rows if row.get("exit_code") not in (0, None))
    write_scheduler_state(
        args,
        scheduler_run_id,
        started_at,
        "failed" if failures else "complete",
        capacity,
        used,
        pending,
        running,
        completed_rows,
        skipped_rows,
        message="completed with %d failed rows" % failures if failures else "complete",
    )
    print(
        "Wrote %d completed rows to %s and %s"
        % (
            len(completed_rows) + len(skipped_rows),
            args.results_json,
            args.results_csv,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
