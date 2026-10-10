# Cycle 3: provenance correction and a recovered no-gain delegation study

Status: PARTIAL historical research. Current-head independent_review=NOT_RUN;
performance=NOT_COMPARABLE to today's incumbent. No live inference, model child,
paid fallback, account activation, routing change, or deployment occurred.

[Assignment](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094586102)
· [cycle-3 claim](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094813718)
· base `0391c13b4bef1e97233e9a064a5b8ba247a05754`.

## Review feedback first: correct the cycle-1 source claim

[Coordinator review](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094788975)
replicated cycle 1 at `2e9b4be4df2f6a7cf6bd867a4bd76b8d5dc0f2a3` and correctly
identified a provenance overclaim. Only comparison.json is committed at the
stated z0intelligence revision; raw.jsonl is an untracked local export. This
cycle checked `git ls-tree`, `git cat-file -e` (raw absent, exit128), committed
comparison bytes and raw SHA-256. The producer checkout has local modifications.

The README and machine receipt now separate comparison_revision from UNKNOWN
raw_producer_revision. A retained digest proves byte integrity, not producer
revision. The existing analytical script also pins the comparison digest: the
baseline accepted a byte-modified comparison with unchanged JSON semantics;
the corrected script refuses it with `Wrong frozen comparison`. Model aggregates
are unchanged. This is a source-binding correction, not a repeated fixture
experiment or new efficacy result. Coordinator replication applies only to the
reviewed prior head, not this correction or the new study below.

## Newly recovered historical study, not new provider execution

Bounded ctx source lookup led to the retained 2026-09-27 Cerebras reasoning-none
study: two report-related Python implementation specifications (summary and
evidence gate), repeated30 times each. These are60 matched pairs, NOT60
independent natural tasks. The study itself says the two fixed task sizes cannot
identify a delegation threshold. No task text, answers, request IDs, account
identifiers, or private paths are exported here.

The source directory has an unborn Git repository, so historical producer SHA
is UNKNOWN. The original protocol freezes18 source-file hashes; all18 still
matched when its unchanged finalize.py was replayed. Content attestation is not
a Git attestation. Four input digests and the concatenated sorted pair-byte
digest are frozen in [delegation60-audit.json](delegation60-audit.json).

Original scripts, executed unchanged with private staged outputs only:

| Script | SHA-256 | Replay |
|---|---|---|
| cache_analysis.py | `838a2e8c24a10f96f827ace2fad2f886de1d14e0bb5b7108de21f0842f65a68b` | every retained result field matches except observation time |
| finalize.py | `a45e3f880a30c0bbeea0e8fd12969c84275722877ae006c4bb6021a358974ce2` | reconciliation.json reproduced byte-for-byte |

The original finalizer checks all frozen source hashes, canonical-byte membership
in the authority ledger, matched worker-token totals, and actual worker output
in every B-arm parent prompt. Neither script performs inference or grades code.
The first finalize invocation correctly failed because the private staging area
omitted complete.json. The test driver incorrectly expected source drift; that
assumption was disproved. After staging the required original receipt/delivery
files, replay passed. No source files were changed to make it pass.

## Identity and accounting result

The360 canonical rows are lifecycle records, NOT360 requests. Applying the
original latest-trace projection gives60 physical attempts and60 distinct native
provider response IDs, all chatcmpl-UUID. Every attempt is recorded completed,
Cerebras qwen-3.8-27b, reasoning_effort=none. No duplicate response identities were
found. All physical receipt cost fields are null. The protocol explicitly has
free_only=false, credit_balance_measured=false and billing class
trial_or_existing_credit_unverified. Historical authorization is not authority
for this lane; this cohort does NOT prove free entitlement today.

| Observed tokens | Parent-only A | Delegated B parent | B worker |
|---|---:|---:|---:|
| Input including cached |1,210,374|1,257,130|15,960|
| Cached input |884,480|861,056|not asserted here|
| Uncached parent input |325,894|396,074|not asserted here|
| Output |32,681|31,065|44,220|
| Reasoning output counter, not added again |6,106|3,535|not asserted here|

Raw parent uncached-input-plus-output increases68,564 tokens. Adding the60,180
observed worker input/output tokens gives128,744 extra observed tokens under
that accounting convention. This is NOT a dollar estimate or full-cost result:
cache billing, worker token inclusion semantics, all coordinator/verifier costs,
invoice truth and host energy remain incomplete. The alternative cache-neutral
parent delta45,140 is an algebraic sensitivity result, not measured savings.

The original unchanged cache analysis yields a decisive negative result on its
measured equal-cache subsets:

| Task specification | Matched repeats | Equal-cache pairs | Mean parent token delta B minus A |
|---|---:|---:|---:|
| Report summary implementation |30|12|+522.0833|
| Evidence-gate implementation |30|8|+1008.125|

Equal cache means identical cached-token counts, not randomized cache exposure.
Positive delta means MORE parent work even before worker tokens. The original
study reports extra paired wall time, but this cycle does not promote old timing
into a current latency benchmark. Provider/backend revision is unknown; matching
response-ID shape alone does not make historical and current arms comparable.

## Outcome conflicts retained, not silently regraded

Original reconciliation records A60/60, B58/60. A later separate grading-v2
summary reports60/60 for both and the worker, after allowing as_integer_ratio.
The final decision embeds both corrected top-level quality and stale nested
B58/60. They are different grading revisions, not different inference runs.
This cycle neither reads protected gold nor runs/changes a grader: these are
historically REPORTED outcomes, not newly independently accepted development
successes. Review must bind the exact grading version before using quality.
Regardless of that correction, both equal-cache token deltas remain negative
for offload benefit. The historical decision kept automatic delegation
ineligible; this lane makes no new production decision or KEEP claim.

## Controls, reuse and limits

[delegation60_audit.py](delegation60_audit.py) is a bounded analytical projection,
not another transcript parser, gateway, quota store or evaluator. It independently
checks frozen bytes, exact group identities, ledger membership, request uniqueness,
reported token arithmetic and agreement with the existing cache report. Run:

    python delegation60_audit.py PRIVATE_COHORT EXISTING_AUTHORITY_LEDGER

The emitted JSON matched the sanitized artifact exactly. Negative controls reject
modified pair bytes, duplicated canonical rows, and an unrelated empty ledger.
The corrected cycle-1 artifact also matches its script output exactly. All these
checks passed; no protected evaluator was edited. Running Python with -O would
disable assertions and is not a supported verification command.

The strongest existing no-model option for report arithmetic is the retained
analyzer itself, which already answers the token/cache question without a worker.
That is a reuse opportunity, not a matched baseline for the historical coding
exercise: do not relabel analysis reuse as measured model-token savings. The
historical frontier parent was gpt-6-astra/low; no fresh same-task frontier arm ran.
Groq and Cerebras current challenger arms remain refused on unproven admission.

Two hypotheses are now more narrowly grounded:
1. Extracting small typed report fields from existing receipts, ONLY when the
   existing deterministic reader cannot answer them; deterministic arithmetic,
   provenance and status joins should stay no-model.
2. Advisory evidence/test-result classification with immutable source IDs and
   an unchanged owning verifier. Generated evidence-gate code must never become
   permission to dispatch; real authority remains with the existing parent.

These are candidate future task families, not a completed mine of naturally
recurring development tasks. Do not rerun the two-specification experiment to
manufacture a size threshold or sample size. Reject stale/missing evidence,
unsafe mutations, guessed tool arguments, billing uncertainty and learned
permission decisions before considering any provider.

## Search boundary, next useful leaf and credit

The distinct Qwen7/11 rate-limit series and original Cerebras402 response remain
unrecovered after bounded ctx/retained-receipt searches. A retrieved third-party
402 troubleshooting page is documentation, not account evidence. Similar
7/11 local-model accuracy rows and unrelated402 substrings are not matches.
Do not combine the newly found60-call cohort with the original120-row smoke
cohort or the140 Groq/Cerebras fixture rows.

Next independently useful slice: audit the retained Groq report-task cohort's
stop/incomplete accounting and the separate four-call default-reasoning Cerebras
cohort, without rerunning inference. Their locators are retained privately. This
can distinguish bounded-output failures from provider-denial evidence before
selecting actual recurring natural task groups. Current free entitlement plus
Kerdoios governed admission and an exact-head independent verifier remain gates
for any live experiment. No new Shadow Network component is needed.

Kevin/kvnloo supplied the mission. The earlier Codex session supplied the retained
study/analysis; original script and z0intelligence/Kerdoios/Tokenomics authors
retain implementation credit. Hermes secondary recovered and replayed evidence;
hermes-coordinator-review supplied the cycle-1 provenance finding. Independent
review of this head and sealed z0evals efficacy acceptance are NOT_RUN.
Coordinator subscription usage is nonzero and recorded privately; invoice cost
unknown. This receipt closes only cycle3's bounded slice, not the host shift.
