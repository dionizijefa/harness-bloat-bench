#!/usr/bin/env python3
"""Compatibility wrapper for the resource-aware scheduler CLI."""

import sys

from harness_bloat_bench.cli.schedule_harbor_tasks import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
