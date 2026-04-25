#!/usr/bin/env bash
set -euo pipefail

echo "Starting verification..."

./scripts/setup.sh
./scripts/lint.sh
./scripts/test.sh

echo "All checks passed."