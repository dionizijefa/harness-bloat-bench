#!/usr/bin/env bash
set -euo pipefail

# Pin the core dataset for repeatable presetup; registry "head" can move.
DATASET="${TERMINAL_BENCH_DATASET:-terminal-bench-core==0.1.1}"
OUTPUT_DIR="${TERMINAL_BENCH_DATA_DIR:-/home/dionizije/dev/terminal-bench-data}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/download_terminal_bench_data.sh [options]

Download a Terminal-Bench dataset for local presetup.

Options:
  --dataset DATASET              Dataset name or name==version.
                                 Default: terminal-bench-core==0.1.1
  --output-dir DIR               Destination directory.
                                 Default: /home/dionizije/dev/terminal-bench-data
  --registry-url URL             Registry URL passed to tb.
  --local-registry-path PATH     Local registry JSON passed to tb.
  --overwrite                    Overwrite an existing non-head dataset.
  -h, --help                     Show this help.

Environment:
  TERMINAL_BENCH_DATASET         Default dataset override.
  TERMINAL_BENCH_DATA_DIR        Default output directory override.
EOF
}

require_value() {
  if [[ $# -lt 2 || "${2-}" == --* ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      require_value "$@"
      DATASET="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$@"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --registry-url)
      require_value "$@"
      EXTRA_ARGS+=("--registry-url" "$2")
      shift 2
      ;;
    --local-registry-path)
      require_value "$@"
      EXTRA_ARGS+=("--local-registry-path" "$2")
      shift 2
      ;;
    --overwrite)
      EXTRA_ARGS+=("--overwrite")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$OUTPUT_DIR")"

if command -v uv >/dev/null 2>&1; then
  TB_CMD=(uv run tb)
else
  TB_CMD=(tb)
fi

"${TB_CMD[@]}" datasets download \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
