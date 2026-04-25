#!/usr/bin/env bash
set -euo pipefail

echo "Setting up project..."

# Python setup
if [ -f "requirements.txt" ]; then
  echo "Installing Python dependencies..."
  python -m pip install --upgrade pip
  pip install -r requirements.txt
fi

if [ -f "pyproject.toml" ]; then
  echo "Installing Python project..."
  pip install -e ".[dev]" || pip install -e .
fi

# Node setup
if [ -f "package.json" ]; then
  echo "Installing Node dependencies..."
  npm ci
fi

echo "Setup complete."