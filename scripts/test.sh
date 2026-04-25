#!/usr/bin/env bash
set -euo pipefail

echo "Running tests..."

# Python tests
if command -v pytest &> /dev/null; then
  pytest
fi

# Node tests
if [ -f "package.json" ]; then
  npm test
fi

echo "Tests complete."