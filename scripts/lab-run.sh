#!/usr/bin/env bash
# Full FlyForge experiment loop. Artifacts land in runs/p0/ (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
RUN_DIR="${RUN_DIR:-runs/p0}"
LEVEL="${LEVEL:-1}"
EVOLVE_GENERATIONS="${EVOLVE_GENERATIONS:-2}"
DAGGER_ROUNDS="${DAGGER_ROUNDS:-2}"
FRESH="${FRESH:-0}"

if ! "$PYTHON" -c "import evolution_lab" 2>/dev/null; then
  pip install -e . -q
fi

if ! "$PYTHON" -c "from evolution_lab.splits import splits_exist; import sys; sys.exit(0 if splits_exist() else 1)"; then
  echo "lab-run: locking splits under data/p0/"
  "$PYTHON" -m evolution_lab lock-splits
fi

echo "lab-run: verify"
bash scripts/verify.sh

ARGS=(lab --run-dir "$RUN_DIR" --level "$LEVEL" --evolve-generations "$EVOLVE_GENERATIONS" --dagger-rounds "$DAGGER_ROUNDS")
if [[ "$FRESH" == "1" ]]; then
  ARGS+=(--fresh)
fi

echo "lab-run: ${ARGS[*]}"
"$PYTHON" -m evolution_lab "${ARGS[@]}"
echo "lab-run: summary at $RUN_DIR/experiment_summary.json"
