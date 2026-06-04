#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier

reward=0
if [[ -f /app/ls-output.txt && -f /app/example.html ]] \
    && grep -q '^alpha-sentinel.txt$' /app/ls-output.txt \
    && grep -q '^beta-sentinel.txt$' /app/ls-output.txt \
    && grep -qi 'Example Domain' /app/example.html; then
  reward=1
fi

printf '%s\n' "$reward" > /logs/verifier/reward.txt
