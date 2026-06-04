#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier

reward=0
if [[ -f /app/long-listing.txt && -f /app/example-headers.txt ]] \
    && grep -Eq 'header-sentinel\.txt$' /app/long-listing.txt \
    && grep -Eq 'listing-target$' /app/long-listing.txt \
    && grep -Eqi '^HTTP/[0-9.]+ ' /app/example-headers.txt; then
  reward=1
fi

printf '%s\n' "$reward" > /logs/verifier/reward.txt
