#!/usr/bin/env bash
set -euo pipefail

echo "Running tests..."

# Python tests
if ! command -v pytest &> /dev/null; then
  echo "Error: pytest is required but not installed." >&2
  exit 1
fi
pytest

# Node tests
if [ -f "package.json" ]; then
  npm test
fi

echo "Tests complete."