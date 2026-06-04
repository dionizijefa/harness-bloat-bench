"""Compatibility wrapper for the web UI server."""

from harness_bloat_bench.ui.server import main


if __name__ == "__main__":
    raise SystemExit(main())
