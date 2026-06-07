#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PORT="${PORT:-8000}"
exec python main.py
