# Secondary offload — cycle 1, 2026-10-10

Continuation: [cycle 2 recovered the original 120-row cohort](historical120.md).
That receipt supersedes the unrecovered-lead status below, not this separate
28-fixture audit or the current entitlement refusal. Historical raw outcomes and
zero costs were weaker than reported; no live admission or promotion follows.

Status: PARTIAL research evidence; independent_review=NOT_RUN. Performance verdict:
NOT_COMPARABLE. No Groq/Cerebras inference, paid fallback, service activation,
production edit, upstream publication, or routing promotion occurred.

Assignment: [EL23 secondary lane](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094586102).
[Claim](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094645442).
Frozen Evolution Lab base/current nightly: `0212fee31a78cdba506f4d2684813962bfe0efb0`.
Only this experiment directory changed. Human objective and authorization: Kevin
(kvnloo). Research/replay: Hermes secondary, gpt-6-astra, existing subscription.
Existing cache/quota code and evaluator labels belong to their original Kerdoios
and z0intelligence contributors; neither implementation was copied or modified.

## Decisive result: refuse live inference, reuse the existing control

Actual inventory cache is dated 2026-10-04T06:34:22.810539+00:00. Existing Kerdoios
`load_inventory_cache()` returns None at the observation time: its six-hour TTL
has expired. The separate canonical capacity export is absent. Neither condition
means a zero balance. Historical diagnostic loading with `ignore_ttl=True` finds
five Groq/Cerebras offers; every dimensional quota is unknown. Existing
`free_only_rejection(PlanningPolicy(free_only=True, no_paid_spill=True))` rejects
all five. Diagnostic TTL bypass is NOT used for execution/admission.

The existing reason text is `free_only: quota window exhausted`, including when
all windows are unknown. Preserve the returned string, but do not report it as
an observed exhausted account. No current account entitlement, credit expiry,
account-scope hash, or governed live admission has been established. Refuse all
five for both candidate task families below. Keys/catalog presence is not proof.

[refusal.json](refusal.json) records the actual replay and opaque cache digest.
[refusal_probe.py](refusal_probe.py) imports unchanged Kerdoios rather than adding
another admission system. Reproduce with PYTHONPATH pointing to the separate
Kerdoios checkout and KERDOIOS_CACHE pointing to the original cache directory:

    python docs/experiments/secondary-offload-20261010/refusal_probe.py

Tested Kerdoios: `9a74ed073c9a4721ae55e9cd7b058dabfe444ce4`. Fresh default head:
`cdb43b05a328a50649d5120bffcdecea09492219`. The four relevant source files are
byte-identical at both revisions; [source-manifest.json](source-manifest.json)
records hashes. The entire newer Kerdoios checkout was NOT tested.

## Current official provider truth, not account admission

Retrieved 2026-10-10; exact URLs/digests in source-manifest.json. HTML Groq fetches
returned 403; official `.md` endpoints succeeded. That documentation retrieval
failure was not an inference API or account failure.

| Literal brief model | Separately observed provider model ID | Documentation only | Account entitlement / live admission |
|---|---|---|---|
| Groq gpt-oss-20b | openai/gpt-oss-20b | listed; rate table 30 RPM, 1K RPD, 8K TPM, 200K TPD | unknown / refused |
| Groq gpt-oss-120b | openai/gpt-oss-120b | listed; same rate table | unknown / refused |
| Groq Qwen 3.8 27B | qwen/qwen3.8-27b | preview; same rate table | unknown / refused |
| Cerebras gpt-oss-120b | gpt-oss-120b | listed; trial 5 RPM, 30K uncached TPM, 90K total TPM, 1M TPH/TPD | unknown / refused |
| Cerebras qwen-3.8-27b | qwen-3.8-27b | listed; same trial limits | unknown / refused |

Groq remaining-token headers refer to TPM, remaining-request headers to RPD;
account limits can differ. Cached tokens do not count toward published rate
limits. Some organizations have separate input/output minute limits. The fetched
Markdown table has ambiguous tier prose (calls its limits Developer while the
models page lists higher Developer ceilings); do not resolve this by assuming
an account tier. Use current account metadata before any execution.

Cerebras explicitly says there is NO permanently free tier: $5 credits expire
30 days after grant, require a verified payment method, and do not renew.
Uncached and total token buckets are independent and refill continuously, not
at fixed daily reset boundaries. Project limits may constrain organization
limits. Trial activation/payment-method changes are outside this assignment.
The current catalog contains the two brief models; `zai-glm-4.7` was deprecated
2026-08-17. Kerdoios#51's older model discussion is not current entitlement.
Groq deprecated `qwen/qwen3.6-27b` and `groq/compound*`; do not silently substitute
old identifiers or transport old account observations into this run.

## Historical inventory: a different cohort recovered, not the 120-call lead

Recovered immutable z0intelligence commit
[`78869d977e847f8f87f3b8806cdcd6ce44b6bc22`](https://github.com/kvnloo/z0intelligence/commit/78869d977e847f8f87f3b8806cdcd6ce44b6bc22),
`results/local-cognition/20260922T-api-cohort/{raw.jsonl,comparison.json}`.
This has 168 evaluator rows across six models, including DeepSeek. The five
Groq/Cerebras models account for 140 rows, 28 per model, on identical fixture IDs.
It is NOT evidence of 120 physical calls or 24 calls per pair. DeepSeek is not
part of this lane and was not executed. No protected fixture bodies/gold labels
were copied into this experiment; only existing aggregate outcomes were read.

| Archived backend label | Schema-valid / 28 | Trajectory-correct / 28 | Reported transport failures | Reported retries |
|---|---:|---:|---:|---:|
| groq/openai/gpt-oss-20b | 27 | 25 | 0 | 0 |
| groq/openai/gpt-oss-120b | 5 | 8 | 23 | 144 |
| groq/qwen/qwen3.8-27b | 23 | 19 | 5 | 30 |
| cerebras/gpt-oss-120b | 6 | 9 | 22 | 132 |
| cerebras/qwen-3.8-27b | 28 | 24 | 0 | 0 |

Recomputed raw-label counts equal the existing report. Each model includes two
compiler-only deterministic solutions. Raw rows have no native request IDs or
input-token fields, and every raw `error` is null despite transport failures in
the companion report. Consequently, physical deduplication, request attempts,
HTTP-status taxonomy and full economics cannot be reconstructed from these
rows alone. Reported retries are wrapper counts, not verified unique requests.
No provider p50/p95 claim is made: raw latency can mix compiler decisions,
failed/retried requests and successful calls. Response-ID family is unavailable.

Baseline failure: interpreting no raw errors or valid syntax as task success
would erase transport failures and overstate utility. Real negative example:
Cerebras Qwen has 28/28 schema-valid but 24/28 trajectory-correct, still only
fixture-level evaluation, not verified development work. Simplest control:
retain the existing compiler's two deterministic solutions; no model is needed
for those decisions. This is not a newly measured savings claim.

[cohort_audit.py](cohort_audit.py) is a bounded stdlib projection, not a new grader,
transcript parser or authoritative event store. It verifies the frozen raw digest,
matched fixture groups and agreement with existing aggregate labels:

    python docs/experiments/secondary-offload-20261010/cohort_audit.py COHORT_DIRECTORY

Result: [cohort-audit.json](cohort-audit.json). Original 120-call/12-of-12 tool-check
lead remains REPORTED/unrecovered. The 8738-token/8K TPM lead was located in a
secondary historical summary, not its original provider response; 413, Qwen
7/11 limits and Cerebras 402/403 therefore remain unverified leads, not new counts.
ctx lexical search worked with the physical retained root and refresh off;
no import/index/repair or broad personal-history export was performed.

## Two next hypotheses and no-go cases

These are hypotheses from recovered control/evaluator families, NOT a completed
mine of recurring natural developer tasks:

1. Tiny typed tool/menu choice, after deterministic legal-action compilation.
   Compare existing compiler/rule control to a short worker suggestion whose
   validity is checked by the unchanged owner. Reject unsafe actions, missing or
   stale evidence, malformed tool arguments and unsupported action IDs. Never
   delegate authority, uncertain mutation, publication, or critical decisions.
2. Short evidence/test-result classification with cited source identifiers and
   deterministic verification. Keep parent judgment and actual verification;
   avoid remote summarization when extracting the known field already answers
   the question. Freeze natural work-item groups before evaluating a model.

A = existing no-model controls replayed here. B = no matched current frontier
arm executed. C/D = refused on actual stale/unknown admission evidence. Therefore
no four-arm quality, latency or cost comparison exists. No promotion warranted.
The next smallest Shadow Network integration is an evidence-only handoff using
existing DecisionOpportunity semantic IDs, authored authority and dispatch;
join Kerdoios admission and Tokenomics to the unchanged z0evals#58 collector.
Do not add a router. First recover original request receipts and current
account truth; then queue a specific exact-head independent review on EL23.

## Verification and accounting

Commands run at unchanged Evolution Lab base plus these docs-only scripts:

- Existing `unittest discover -s tests -p test_capacity_queue.py`: 29 passed,
  exit 0 using an existing NumPy environment. Initial default Python run failed
  importing NumPy; retained as an environment failure.
- Existing Kerdoios `unittest discover -s tests -p 'test_quota_*.py'`: 44 passed,
  exit 0, PYTHONDONTWRITEBYTECODE=1, read-only separate checkout.
- Both analytical scripts: exit 0; cache denied, all five rows refused, frozen
  cohort digest/group/count assertions passed.
- Full Evolution Lab suite in available environment: 256 tests, 8 errors,
  2 skips, exit 1; all errors are missing optional JAX. No dependency install or
  peer environment edit. This is NOT a full-suite GREEN claim.
- `python -m evolution_lab gym-smoke --n 8`: exit 0, teacher perfect.
- No evaluator changed; owning tests passing are scoped invariants, not an
  independent offload efficacy verdict. Reviewer identity: none; NOT_RUN.

Tokenomics semantics inspected at `3fa33ee83d02e43a37d7890656a4151e200f2275`:
incremental vs aggregate attribution and root/worker/verifier roles remain
separate; negative outcomes take precedence. Its verified-only economics cannot
stand in for full-attempt cost, and a numeric default zero is not invoice proof.
Native coordinator metadata is retained privately, including subscription mode,
input/output/cache/reasoning tokens and API-call count. During execution it was
incomplete; actual invoice cost was null. Zero Groq/Cerebras calls does NOT mean
zero coordinator cost/tokens. Host joules and end-to-end savings remain unknown.

Next cycle: recover the original 120-call cohort using native source identities,
not more repetitions of this 28-fixture proof. If inaccessible, independently
characterize missing-request-ID/retry coverage using existing receipt consumers.
All raw histories remain private. Scope and independent-review gaps stay open.
