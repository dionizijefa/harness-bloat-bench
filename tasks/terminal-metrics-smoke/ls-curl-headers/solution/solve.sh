#!/usr/bin/env bash
set -euo pipefail

cd /app
ls -la > /app/long-listing.txt
curl -I -L https://example.com > /app/example-headers.txt
