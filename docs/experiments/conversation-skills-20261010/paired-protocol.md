# Frozen producer handoff: independent paired experiment

Owner: Evolution Lab #23 / EL-23-S. Judge: z0evals #93 / existing draft PR95.
The adjacent `handoff.json`, `run-002.json` and `sampler-replay.json` contain only
approved metadata. This is READY for identity review, BLOCKED for actual skill
execution. It is not privacy clearance, a gold label, an efficacy result or KEEP.

## Immutable producer identity

Recheck the private artifacts with `scripts/conversation_skills_receipt.py` and
use the frozen receipt and lineage hashes in `handoff.json`. Obtain private
artifact access through the existing owner, not by posting draft text or source
exports. Keep the external executable at its clean pinned ref; preserve Joshua
Warren, Umang Bhalla and the external project's attribution and unresolved rights.
Any failed generator, missing/changed draft, changed corpus/ref/receipt, partial
mapping or ambiguous native-session grouping must refuse the handoff.

The original sampler was reexecuted twice, unchanged, with seed23, maximum2
conversations and budget12000. It reproduced one selected record and a 12000-char
excerpt. The excerpt is truncated and is not complete rendered JSON. Its hash
and the selected full record hash are frozen, but the original on-wire model
request was not captured: the request hash is explicitly reconstructed, not
attested historical wire data. No new inference was used for this replay.
Do not extrapolate candidate correctness from an excerpt or generator formatting.

All nine input sessions, including branches/retries/continuations of their source
qualification family, are conservatively treated as exposed development data.
The group hash is a quarantine identity, not an invented native work-item ID.
Do not split exported leaves across development/confirm/future, count repeated
prefixes as independent tasks, or use any exposed member as held-out proof.
The private mapping is a producer assertion requiring independent source checks.

## Existing independent interface, not a second judge

Pinned verifier: `kvnloo/z0evals` at
`ffa59661aaee9723af408c1f34e4e6207ab4f643`. Read its existing
`studies/protected-evaluator-v0/README.md` and trusted manifest before execution.
The judge, not this producer, freezes labels/cohorts and resolves protected roots.
Its actual CLI shape is:

```sh
python scripts/protected_eval.py --manifest MANIFEST --state-dir PRIVATE_STATE \
  score --cohort COHORT --predictions CANDIDATE_PREDICTIONS \
  --candidate-id CANDIDATE_ID --candidate-revision FULL_PRODUCER_GIT_SHA \
  --baseline-predictions BASELINE_PREDICTIONS --baseline-revision FULL_BASELINE_GIT_SHA
```

Each private prediction JSONL must use the verifier's existing ID/prediction
contract and cover exactly the same frozen cohort. The scorer checks complete
vectors, group-macro exact match, cross-cohort lineage, contamination, sealed
adaptive manifest/cohort identity and query budget. One paired comparison uses
one query; ties/regressions are DISCARD even above an absolute threshold.
Its aggregate result hashes and verdict are the independent scoring evidence.
Do not alter its schema/labels/state, query `future`, tune from protected labels,
or generate predicted answers here. Supplied vectors are not proof of native
agent execution, safety, repeatability or adoption approval. Earlier PR95 receipt
failure remains historical evidence even though a later receipt check passed.

## Conditions to freeze before any permitted trial

The verifier and authorized native-run owner must pin the same task/cohort,
model and weight/version, system/tool configuration, tool permissions, context,
starting filesystem/cache conditions, stopping budget and repeat/seed policy for
both arms. The baseline has no candidate instructions; the candidate arm loads
exactly one reviewed content hash, explicitly, without persistent installation.
Two drafts require distinct comparisons: do not combine them after seeing scores.
Record any deterministic/prior-skill comparator separately rather than quietly
changing the no-skill baseline. Generation model identity is not trial-model proof.
Freeze cohort/manifest/configuration digests and comparison revisions before
running, and preserve native input/output evidence privately.

Count actual calls, not assistant messages containing calls. Preserve valid,
bad/incorrect-argument and unnecessary/redundant calls separately, alongside
errors, user corrections, retries and task success. The independent judge must
supply these classifications from observed behavior; model-generated claims and
exported prose are not labels. Record native wall time, cached/uncached/reasoning
usage and actual cost only where instrumentation supports them. Missing values
stay null, not zero. Local API cost for the original generation was zero, but
trial cost and host joules are unmeasured. Compare latency only inside compatible
response-ID strata; preserve the existing two-upstream correction on #137.

## Blockers and prohibited shortcuts

Neither draft has been independently inspected against source facts or cleared
for privacy/rights. Unreviewed bodies must not reach a frontier/remote inference
endpoint. The corpus omits event identity/timing/usage/details; source-native/ctx
joins remain necessary, and current ctx root/config availability is blocked.
No trusted disjoint held-out cohort, paired native run, complete predictions or
independent result exists. This session may not create another agent worker or
execution harness to substitute for those missing prerequisites.

Retain `installed=false`, `promotion_eligible=false`, all null outcomes and the
BLOCKED status until the existing owners supply actual evidence and required
approvals. Do not install drafts, train a model, redistribute unresolved-rights
source/draft text, activate runtime consumers or manufacture a synthetic gain.
