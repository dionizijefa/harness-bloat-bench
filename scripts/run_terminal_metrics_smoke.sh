#!/usr/bin/env bash
set -euo pipefail

TASK_IDS="${TASK_IDS:-ls-curl-example,ls-curl-headers}"
MODEL="${MODEL:-qwen/qwen3.7-max}"
HARNESS_VERSION="${HARNESS_VERSION:-latest}"
DATASET_PATH="${DATASET_PATH:-tasks/terminal-metrics-smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/terminal_metrics_smoke}"
LIVE=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_terminal_metrics_smoke.sh [--live] [-- TASK_ARGS...]

Run the local Terminal-Bench/Harbor smoke tasks that ask the model to run ls
and curl. Without --live, this only runs the matrix runner with --dry-run.

Defaults:
  TASK_IDS=ls-curl-example,ls-curl-headers
  MODEL=qwen/qwen3.7-max
  HARNESS_VERSION=latest
  DATASET_PATH=tasks/terminal-metrics-smoke
  OUTPUT_DIR=outputs/terminal_metrics_smoke

Options:
  --live      Actually call OpenRouter and Harbor.
  -h, --help  Show this help.

Environment:
  OPENROUTER_API_KEY is required for --live.
  TASK_IDS, MODEL, HARNESS_VERSION, DATASET_PATH, and OUTPUT_DIR override defaults.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live)
      LIVE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$LIVE" -eq 1 && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required for --live" >&2
  exit 2
fi

CMD=(uv run python -m harness_bloat_bench.cli.run_matrix \
  --dataset-path "$DATASET_PATH" \
  --task-ids "$TASK_IDS" \
  --harness codex-cli \
  --harness-versions "$HARNESS_VERSION" \
  --models "$MODEL" \
  --n-attempts 1 \
  --n-concurrent 1 \
  --output-dir "$OUTPUT_DIR" \
  --tbench-runs-dir "$OUTPUT_DIR/tbench_runs" \
  --results-json "$OUTPUT_DIR/results.json" \
  --results-jsonl "$OUTPUT_DIR/results.jsonl" \
  --results-csv "$OUTPUT_DIR/results.csv")

if [[ "$LIVE" -eq 0 ]]; then
  CMD+=(--dry-run)
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"
