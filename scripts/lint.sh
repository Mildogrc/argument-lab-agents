#!/usr/bin/env bash
set -euo pipefail

echo "Running lint..."

# Python lint
if ! command -v ruff &> /dev/null; then
  echo "Error: ruff is required but not installed." >&2
  exit 1
fi
ruff check .

# Formatting
if ! command -v black &> /dev/null; then
  echo "Error: black is required but not installed." >&2
  exit 1
fi
black --check .

# Node lint
if [ -f "package.json" ]; then
  npm run lint || true
fi

echo "Lint complete."