#!/usr/bin/env bash
# FlyForge autoresearch entry (bounded kernel tuning loop)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TAG="${TAG:-$(date +%Y%m%d)}"
exec python3 -m evolution_lab autoresearch --tag "$TAG" "$@"
