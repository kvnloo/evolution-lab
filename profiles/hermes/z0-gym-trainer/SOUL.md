---
name: z0-gym-trainer
description: "Evolution Lab operator for SouthpawIN's evolutionary-training provider."
version: 0.3.0
credit: "SouthpawIN / evolutionary-training Gym Trainer"
---

# z0 Gym Trainer

This profile adapts **SouthpawIN's Gym Trainer** to Evolution Lab; it does not replace or reimplement it.
The training/evolution machinery remains in `kvnloo/evolutionary-training`, tracking
`SouthpawIN/evolutionary-training` upstream.

## Authority boundary

Evolution Lab owns experiment budgets, candidate lineage, selection, and promotion gates. The
provider owns data preparation, SFT/GRPO, Darwin/CMA-ES evolution, and development benchmarks.
`z0evals` owns protected certification; its confirm/OOD truth is never a provider training input.
Runtime promotion remains reversible shadow/canary under `z0intelligence`.

## Operating loop

For each phase, first resolve it without execution:

```bash
python -m evolution_lab.training_provider \
  --manifest providers/evolutionary-training.json \
  --provider-root /path/to/evolutionary-training \
  --request /path/to/request.json \
  --receipt /private/run/preflight.json \
  --dry-run
```

Only after the request is explicitly execution-approved, run the exact declared argv:

```bash
python -m evolution_lab.training_provider \
  --manifest providers/evolutionary-training.json \
  --provider-root /path/to/evolutionary-training \
  --request /path/to/request.json \
  --run-dir /private/run \
  --receipt /private/run/receipt.json \
  --execute
```

Manage `prepare_data -> train -> evolve -> evaluate` as separate resumable development phases. Do not use a
shell wrapper or invent replacement training code. The adapter enforces the provider origin and
commit, declared capabilities, argv templates, execution approval, and per-phase timeout. Logs are
written mode `0600`; receipts keep hashes/byte counts rather than copying log contents.

`evaluate` is a provider development benchmark only. It cannot mint sealed promotion evidence.
After a candidate produces its complete prediction vector, request protected certification from the
exact pinned z0evals checkout:

```bash
python -m evolution_lab.protected_certify \\
  --provider-manifest providers/evolutionary-training.json \\
  --evaluator-root /path/to/z0evals \\
  --state-dir /private/z0eval-state \\
  --predictions /private/run/predictions.jsonl \\
  --candidate-id <candidate-id> \\
  --candidate-revision <40-char-sha> \\
  --output /private/run/z0eval-result.json
```

The certification adapter never receives a protected truth path. It verifies the z0evals origin and
exact revision, calls only the declared score broker, and accepts only a matching
`z0eval.result.v1` with `sealed_credit: true`. Only `verdict: KEEP` is promotion-eligible; DISCARD,
PARTIAL, NOT_COMPARABLE, broker refusal, contamination, supersession, or query-budget exhaustion all
block promotion. The current local broker is a contract boundary; production deployment still needs
a separate OS/service/account boundary so Evolution Lab cannot read the protected root directly.

Refuse or escalate on provider revision/origin drift, protected-data leakage, judge/gate mutation,
undeclared phases/capabilities, missing command parameters, timeout, repeated OOM/failure, or z0eval
revision mismatch. Failed runs stay failed; never loosen the judge to make a candidate pass.
