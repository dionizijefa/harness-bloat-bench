# Terminal Metrics Smoke Tasks

These are intentionally small local Terminal-Bench/Harbor tasks for checking
harness accounting and command/event capture across harness versions.

Run them through the matrix runner with:

```sh
uv run python -m harness_bloat_bench.cli.run_matrix \
  --dataset-path tasks/terminal-metrics-smoke \
  --task-ids ls-curl-example,ls-curl-headers \
  --harness codex-cli \
  --harness-versions latest \
  --models qwen/qwen3.7-max \
  --n-attempts 1 \
  --n-concurrent 1
```

The tasks ask the agent to run directory listing commands and `curl
https://example.com`, then save command outputs to files checked by the
verifier.
