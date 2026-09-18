# Evolution Lab ↔ z0intelligence cutover

- **Runtime dependency**: `openjev-phase1` optional extra installs from `kvnloo/z0intelligence` (package name unchanged).
- **Production preflight**: owned by z0intelligence. `evolution_lab preflight` is research/shadow only and never mints `verified_success`.
- **Candidate export**: `evolution-lab export-z0int --candidate … --output …` emits `z0int.candidate_artifact.v1` with `promotion.status=candidate` only. No production registry writes.
