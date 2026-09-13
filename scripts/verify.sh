#!/usr/bin/env bash
# Fail-closed verifier. Mutation is not in the registered table.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "verify: python interpreter not found ($PYTHON)" >&2
  exit 1
fi

"$PYTHON" -m unittest discover -s tests -p 'test_*.py'
"$PYTHON" -m evolution_lab gym-smoke --n 8
echo "verify: mutation n/a (not in registered table)"
echo "verify: ok"
