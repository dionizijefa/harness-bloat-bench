#!/usr/bin/env python3
"""Compatibility wrapper for the matrix runner CLI."""

import sys

from harness_bloat_bench.cli.run_matrix import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
