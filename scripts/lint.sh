#!/usr/bin/env bash
set -euo pipefail

echo "Running lint..."

# Python lint
if command -v ruff &> /dev/null; then
  ruff check .
fi

# Optional: formatting
if command -v black &> /dev/null; then
  black --check .
fi

# Node lint
if [ -f "package.json" ]; then
  npm run lint || true
fi

echo "Lint complete."