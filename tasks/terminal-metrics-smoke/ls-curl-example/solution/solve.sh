#!/usr/bin/env bash
set -euo pipefail

cd /app
ls > /app/ls-output.txt
curl -L https://example.com > /app/example.html
