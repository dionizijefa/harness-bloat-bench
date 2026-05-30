# Driftbench

Minimal experimental loop for running Terminal-Bench tasks across harness versions and models.

Install dependencies:

```bash
uv sync
```

## Runner

Prerequisites for real runs:

- Docker available for Terminal-Bench task containers
- Model credentials expected by the selected Terminal-Bench agent, such as `OPENAI_API_KEY` for `codex-cli`

Example:

```bash
uv run python run_matrix.py \
  --harness codex-cli \
  --harness-versions v1,v2,v3 \
  --models model_a,model_b \
  --tasks terminal-bench-core \
  --repeats 3
```

The initial harness is `codex-cli`. It maps to Terminal-Bench's built-in `codex` installed agent and passes each harness version as:

```bash
--agent codex --agent-kwarg version=<harness_version>
```

Terminal-Bench's Codex installed agent uses that value when installing `@openai/codex@<harness_version>` inside task containers.

Useful options:

```bash
# Run a small subset first.
uv run python run_matrix.py \
  --harness codex-cli \
  --harness-versions 0.135.0 \
  --models gpt-5-codex \
  --tasks terminal-bench-core \
  --task-ids hello-world \
  --repeats 1

# Check the planned matrix without invoking Terminal-Bench.
uv run python run_matrix.py \
  --harness codex-cli \
  --harness-versions 0.135.0 \
  --models gpt-5-codex \
  --tasks terminal-bench-core \
  --task-ids hello-world \
  --dry-run

# Add cost estimates.
uv run python run_matrix.py \
  --harness codex-cli \
  --harness-versions 0.135.0 \
  --models gpt-5-codex \
  --tasks terminal-bench-core \
  --task-ids hello-world \
  --pricing-file pricing.example.json
```

Results are appended to:

- `outputs/results.jsonl`
- `outputs/results.csv`

Terminal-Bench run artifacts and task logs are stored under `outputs/tbench_runs/`. Runner stdout and stderr captures are under `outputs/logs/`.

Each result row includes pass/fail, runtime, stdout/stderr log paths, token counts when Terminal-Bench reports them, estimated cost when a pricing file is supplied, model, harness version, task ID, timestamp, local git commit, Terminal-Bench commit, and the exact command.

## Analysis

Create plots:

```bash
uv run python analysis/plot_results.py --results outputs/results.csv --out-dir outputs/plots
```

The script writes:

- `accuracy.png`
- `cost_per_task.png`
- `tokens_per_task.png`
- `latency.png`
- `accuracy_vs_cost.png`
- `summary_by_harness_model.csv`

There is also a starter notebook at `analysis/example_analysis.ipynb`.

## Adding Harnesses

Add a new `Harness` subclass in `run_matrix.py` and register it in `get_harness()`. Keep each harness responsible only for translating `(version, model)` into Terminal-Bench CLI arguments.
