#!/usr/bin/env bash
set -euo pipefail

TASK_ID="${TASK_ID:-crack-7z-hash}"
MODEL="${MODEL:-qwen/qwen3.7-max}"
HARNESS_VERSION="${HARNESS_VERSION:-latest}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smoke}"
LIVE=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_smoke_task.sh [--live] [-- TASK_ARGS...]

Run the shortest Terminal-Bench 2.1 task through run_matrix.py.

Defaults:
  TASK_ID=crack-7z-hash
  MODEL=qwen/qwen3.7-max
  HARNESS_VERSION=latest
  OUTPUT_DIR=outputs/smoke

Options:
  --live     Actually call OpenRouter. Without this flag, runs --dry-run.
  -h, --help Show this help.

Environment:
  OPENROUTER_API_KEY is required for --live.
  TASK_ID, MODEL, HARNESS_VERSION, and OUTPUT_DIR override defaults.
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

DRY_RUN_ARGS=()
if [[ "$LIVE" -eq 0 ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

uv run python run_matrix.py \
  --harness codex-cli \
  --harness-versions "$HARNESS_VERSION" \
  --models "$MODEL" \
  --task-ids "$TASK_ID" \
  --n-attempts 1 \
  --n-concurrent 1 \
  --output-dir "$OUTPUT_DIR" \
  --tbench-runs-dir "$OUTPUT_DIR/tbench_runs" \
  --results-json "$OUTPUT_DIR/results.json" \
  --results-jsonl "$OUTPUT_DIR/results.jsonl" \
  --results-csv "$OUTPUT_DIR/results.csv" \
  "${DRY_RUN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
