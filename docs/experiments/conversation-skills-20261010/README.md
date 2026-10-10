# Conversation skill mining: first review-only slice

Tracking: [Evolution Lab #23](https://github.com/kvnloo/evolution-lab/issues/23), EL-23-S.
Claim: https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6092665704
Source producer: Claude EL-23-X; independent verifier: z0evals #93 / PR95.
Shared planning source: z0intelligence's `coord/multiharness-strategy-board-20261009` branch.

## What actually ran

The existing `kvnloo/ai-data-extraction` fork was invoked directly, unchanged,
from an isolated detached worktree at `9176c2dd81c8a101b52debd4cec88573284e2b30`.
No extraction or generation code was copied into Evolution Lab.

Input was Claude's existing bounded active-branch OMP export: nine isolated
synthetic qualification sessions, 2,304,611 bytes. Its exact SHA-256 is in the
receipts. The broad personal history store was not read or exported. Sampling
seed 23, maximum two conversations, and a 12,000-character budget actually
selected one of nine records. This is a bounded plumbing experiment, not a
representative corpus or a capability benchmark.

Both invocations used the confirmed local Ollama daemon at
`http://127.0.0.1:11435/v1`, with no API-key header. Existing locally available
model weights were used; no weights were downloaded for this slice.

| Invocation | Model | Exit | Valid draft files |
| --- | --- | --- | --- |
| run-001 | qwen2.5:3b | 1 | 0 |
| run-002 | qwen2.5-coder:7b | 0 | 2 |

The first result was rejected by the original generator: `Model output was not
a valid JSON object`. That negative result was not repaired, substituted or
relabeled. The second invocation produced exactly two `SKILL.md` files, and
the original generator validated their JSON/name/description/body shape.
This does not establish usefulness, safety, privacy clearance or improvement.
Input identity, model digest and private candidate hashes are recorded in the
adjacent JSON receipts. Draft names and text, raw exports and model output are
not public artifacts. No skill was installed.

Original execution shape (PRIVATE_ROOT denotes the private experiment store):

```sh
python "$PRIVATE_ROOT/ai-data-extraction/corpus_to_skills.py" \
  "$PRIVATE_ROOT/input/omp-active-frozen.jsonl" \
  --output-dir "$PRIVATE_ROOT/candidates/run-002" \
  --model qwen2.5-coder:7b --base-url http://127.0.0.1:11435/v1 \
  --api-key-env EL23_LOCAL_NO_KEY --skills 2 \
  --corpus-char-budget 12000 --max-conversations 2 --seed 23 --timeout 180
```

The key variable was explicitly unset. Output is create-only; retry uses a new
output directory rather than overwriting either positive or negative evidence.
The original generator does not expose response token usage or an output-token
cap. Per-call latency, tokens, electricity and joules were not measured and must
not be inferred. No hosted experiment inference was used; the receipts' zero
API cost is local-only, not a claim of zero machine/energy cost.

## Source/provenance limits that remain authoritative

Claude's EL-23-X comparison establishes that the original export omits native
per-entry IDs, timing/reasoning-token/cost fields and structured result details.
`--all-branches` can duplicate an identical shared prefix across leaf exports.
Train/confirm/future grouping must therefore be by native session/work item,
not by exported conversation. Hook context and tool-result roles are additionally
lossy for the tested Claude Code exports; those exports were not added here.
Existing ctx/native records remain the source of truth for source-event joins.
A new index, history database or guessed reconstruction is not an acceptable fix.
The host's reported ctx root/config failure remains a source-coverage blocker.

Preserve the two-upstream latency correction on z0intelligence #137: comparisons
must be stratified by response-ID shape. No latency comparison is made here.
Neither Claude `a90e3be` nor Codex `199a849` was changed or behaviorally requalified.

## Tests and handoff

The original fork's stdlib suite passed 22 tests at its pinned revision.
Evolution Lab's isolated CPU environment passed its existing 271 tests with one
skip and `python -m evolution_lab gym-smoke --n 8`. Those validate their own
interfaces, not the two draft skills. The unprepared environment initially lacked
NumPy, then JAX; those failures remain in local receipts. Only an isolated venv
was prepared, with CPU JAX; no global runtime, model route or installation changed.
Exact Evolution Lab receipt heads and repeated commands belong in the PR/issue
receipt, not a circular self-referential commit field in these JSON artifacts.

The read-only `scripts/conversation_skills_receipt.py` boundary checks the existing
run JSON against the private corpus, candidate files and a clean exact generator
checkout. Supply `--receipt`, `--receipt-sha256`, `--corpus`, `--candidates`, and `--generator`; it
prints only hashes/counts and a review-only identity status, or exits 2 with a
sanitized refusal. It neither copies nor installs artifacts. Receipt declarations
are producer assertions, not signed process attestations; an old success receipt
does not prove a new generator invocation occurred. Recheck the frozen artifacts
at consumption time. Pin the expected receipt hash from a frozen handoff, not a
mutable receipt newly hashed by the consumer. Duplicate JSON keys and additional
unaccounted draft output are refused. Failed/unknown exits cannot reuse prior
draft files. Source groups and independent verification remain separate.

Next eligible producer slices are executable refusal controls, source-group
leakage eligibility and a frozen paired-experiment handoff to the independently
owned z0evals gate. That gate must compare a no-skill baseline and candidate on the
same frozen cohort and retain malformed/unsupported/missing-data controls.
No held-out outcome, KEEP decision, learning gain, token savings, production
adoption, training or curriculum promotion is claimed. z0evals PR94's historical
reporting correction remains separate from fresh candidate qualification.

## Attribution and redistribution boundary

External project: `sybil-solutions/ai-data-extraction`,
used through the existing `kvnloo/ai-data-extraction` fork. Its README credits
Joshua Warren's OMP coverage and Umang Bhalla's Droid contribution; preserve that
credit. The README's license statement does not resolve the absence of an
explicit license file/current GitHub license metadata. Keep the executable
project separate; no vendoring, raw-history publication or draft redistribution
is authorized by this slice.
