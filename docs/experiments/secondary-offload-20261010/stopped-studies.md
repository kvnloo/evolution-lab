# Cycle 4: stopped-study denominators and bounded-output failures

Classification: PARTIAL historical evidence; independent_review=NOT_RUN;
performance=NOT_COMPARABLE. No live provider calls, model subprocess, paid
fallback, new infrastructure, protected grading, or production changes.

[Assignment](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094586102)
· [claim](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094912113)
· base `7acf8b18d1414bfb406cbdaccafe4daf86558843`.

## Baseline discrepancy and unchanged deterministic control

Recovered two separate September 27 report-implementation studies, not the
120-row smoke cohort, 140 provider fixture rows, or 60-pair reasoning-none
study already audited. Each repeats only two specifications (summary and
evidence gate); they are not independent natural developer task populations.

The existing Groq top-level summary is an intermediate snapshot: 5 pairs,
6 physical calls and 10,899 worker tokens. The unchanged original report.py,
replayed on all retained pair files and the existing authority ledger, instead
finds 38 pairs, 38 receipted physical calls and 68,877 worker tokens. This is a
stale-denominator failure, not additional inference performed by this cycle.
The separate later grading-v3 report already contains the larger denominator.
No new report generator, grader, gateway, scheduler or quota store is needed.

Both source reports write outputs despite their "read-only" docstrings. They
were inspected before execution; only protocol files and pair receipts were
staged into private output directories. The original scripts ran unchanged,
reading the existing ledger and writing only staged reports/canonical exports.
The v3 script can launch graders when a grade is missing; it was NOT executed.
No inference-producing runner/worker was executed.

| Existing script | SHA-256 | Result |
|---|---|---|
| Groq report.py | `2fb863322b1d60e370e75633733c78e0adae4f5aa510d0841d31022d61cdcfdc` | exit0; intermediate summary contradicted by full retained data |
| Cerebras report.py | `15207d73d34e9b3826d2c9a2511a666de857da5eacc2f20b32e6f0b830c9d9f1` | exit0; all summary fields reproduced except observation time |

Historical producer Git SHA remains UNKNOWN: the source parent repository has
an unborn main branch, and HEAD verification exits128. All17 Groq and18 Cerebras
protocol-pinned source hashes match. Content attestation is not Git provenance.
Source, pair-byte and worker-byte digests are frozen in
[stopped-audit.json](stopped-audit.json); raw rows, prompts, responses, account
identities and private locators remain private with restrictive permissions.

## Distinguish lifecycle state, physical completion and useful output

| Retained observation | Groq openai/gpt-oss-20b | Cerebras qwen-3.8-27b |
|---|---:|---:|
| Completed A/B pairs |38|4|
| Lifecycle records |230|24|
| Receipted physical attempts / distinct native response IDs |38 /38|4 /4|
| Physical status completed / incomplete |28 /10|0 /4|
| Finish reason stop / length |28 /10|0 /4|
| Observed worker input / output tokens |12,103 /56,774|1,208 /8,192|
| Known provider cost rows |0|0|
| Additional unresolved orchestration records |2|0|

All receipted native response IDs are chatcmpl-UUID and unique within each
provider/model cohort. The original latest-live-trace projection is reused;
multiple lifecycle events are not multiple requests. Canonical exports match
the unchanged replay byte-for-byte and every retained line belongs to the
existing authority ledger. These checks do not certify task quality.

Groq's next logical item has a worker-error receipt: returncode1,
TimeoutError: timed out, status "uncertain; identical identity only". The ledger
retains started/acquired orchestration records without a corresponding physical
receipt. This is NOT proof of a39th provider request, nor proof it consumed zero
tokens, nor a429/quota failure. All-attempt physical and economic coverage is
UNKNOWN. No retry was attempted by this audit. The stop must remain unresolved
rather than disappear from the accounting denominator.

Cerebras' four calls exhausted the2048 output-token limit and returned empty
final worker content, with finish_reason=length. The historical exclusion report
attributes this to high/default reasoning; the protocol does not pin an explicit
reasoning_effort. That attribution is REPORTED, not a newly measured reasoning
counter. These are bounded-output failures, NOT402/403 billing/authorization
denials. All four pairs required repair/fallback according to their retained
labels. Provider output tokens are nonzero despite zero final worker content.
The historical protocol has free_only=false and unmeasured trial/existing credit;
it grants no current free entitlement. Groq's historical free_only=true also
cannot establish current account truth or admission.

## Negative economics and the tempting false win

Both historical parent arms identify gpt-6-astra/low. No matched fresh incumbent
ran, and provider/backend versions are not pinned. A common response-ID shape
cannot establish identical backends. Old timings are retained in the machine
projection, not promoted into a current latency comparison or TTFT measurement.

Groq's whole-cohort uncached-parent-input-plus-output delta is -72,708 tokens.
Adding the68,877 observed worker tokens leaves -3,831. That tempting subtotal
is NOT measured savings: B received87,552 more cached parent input tokens,
stop coverage is incomplete, provider costs are null, cache billing and
coordinator/verifier costs are missing. The algebraic cache-neutral parent
delta is +14,844, a sensitivity calculation rather than a new observation.

The unchanged report's measured zero-cache subsets show MORE parent work:

| Task | Pairs with both arms uncached | Mean parent token delta B minus A |
|---|---:|---:|
| Summary implementation |5|+430.8|
| Evidence-gate implementation |10|+555.3|

These are selected repeated-task subsets, not randomized independent evidence.
They nevertheless falsify interpreting the cache-confounded aggregate as a
robust delegation win. Existing deterministic aggregation answers this accounting
question without another model. It is not a matched no-model code-authoring arm.

Cerebras adds18,340 observed parent tokens and9,400 worker tokens: +27,740 under
the same convention. This is an incomplete token subtotal, not full dollar cost.
No claim of free inference, invoice savings or host-energy savings follows.

## Historical grading revisions are not new outcomes

Original Groq pair labels report evidence-gate parent A11/19, B4/19 and worker0/19.
The retained grading-v3 summary reports A19/19, B19/19 and worker0/19. Non-quality
report fields agree exactly. Both versions are preserved; neither is treated as
new independent gold, and no grader was read for labels, modified or run.
The worker gate remains0/19 in both versions despite some completed transport.
Cerebras pair labels report parent A/B4/4 and worker0/4, also historical only.

Verdict: retain negative/refusal evidence; do not promote the bounded default
Cerebras configuration or claim a Groq savings win. PARTIAL recovery,
NOT_COMPARABLE to current service/incumbent; sealed efficacy NOT_RUN.

## Verification and next smallest integration

[stopped_audit.py](stopped_audit.py) is a bounded analytical projection using
stdlib only, not an event store or independent evaluator. Reproduction:

1. Privately stage protocol.json, protocol.sha256 and each */pair.json, preserving
   relative layout, separately for each cohort. No prompts/gold are needed.
2. Run each digest-pinned original report.py with its staged directory argument.
   It reads the existing authority ledger; do not run its sibling runners/graders.
3. Supply original Groq/Cerebras directories, replay root (free-worker-ab and
   cerebras-ab subdirectories), and existing authority ledger:

       python stopped_audit.py GROQ_SOURCE CEREBRAS_SOURCE REPLAY_ROOT LEDGER

4. Compare emitted JSON to stopped-audit.json. Do not use Python -O: assertions
   are the verification mechanism, not a production admission boundary.

Actual checks: both unchanged replays exit0; bounded audit matches the sanitized
artifact; wrong-ledger and modified-protocol controls reject for both cohorts;
stale-summary, unresolved-stop, output-exhaustion and cache-sensitivity checks
pass. Local existing gym-smoke n8 exit0, teacher perfect; git diff --check exit0.
No full-suite GREEN claim is made from these scoped local checks.

No implementation integration is justified by these two repeated specifications.
The smallest future Shadow Network handoff is to preserve existing semantic
work-item/dispatch identities and unresolved-attempt state through Tokenomics
and the existing z0evals#58 collector, with Kerdoios current entitlement/admission.
Do not add a retry or infer zero usage from absence. First queue exact-head
independent review of the38-ID join, uncertain stop and cache-conditioned result.

Next independent leaf: inspect existing DecisionOpportunity/dispatch and
Tokenomics outcome joins against the current collector's required fields,
read-only, to identify the smallest missing evidence seam for actual natural
work-item groups. Do not rerun these stopped studies or prior120/140/60 cohorts.
The distinct Groq Qwen7/11 and original Cerebras402 leads remain unrecovered;
this Groq20b timeout and Cerebras length exhaustion are not substitutes.
Current provider admission remains unproven/refused, so no live experiment.

Kevin/kvnloo supplied mission and scope; earlier study authors and original
z0intelligence/Kerdoios/Tokenomics/report contributors retain implementation
credit. Hermes secondary recovered and replayed evidence. Reviewer: none for
this head. Existing-subscription coordinator usage is nonzero, privately
recorded and incomplete while running; invoice and host joules UNKNOWN.
This concludes only cycle4's slice, not the host-controlled shift.
