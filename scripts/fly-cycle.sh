#!/usr/bin/env bash
# Outer cycle around autoresearch — evolve the production fly until stopped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 -m evolution_lab cycle --forever --skip-unit-tests "$@"
