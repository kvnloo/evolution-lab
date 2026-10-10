# Cycle 2: original 120-row cohort recovered, accounting claims narrowed

Classification: PARTIAL. independent_review=NOT_RUN; performance=NOT_COMPARABLE.
No new provider inference, paid fallback, model subprocess, routing change or
account activation. Historical results do not qualify current admission.

[Assignment](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094586102)
and [cycle-2 claim](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094735930).
Frozen Evolution Lab base: `2e9b4be4df2f6a7cf6bd867a4bd76b8d5dc0f2a3`.
This supersedes only cycle 1's “120-call lead unrecovered” status; its separate
28-fixture audit, current entitlement refusal and negative evidence remain.

## Provenance and actual recovery

Existing ctx located secondary references but did not index the original DSH
session under its native ID. Bounded native filename discovery found the
retained DSH Pi-format export, then original benchmark and Tokenomics files.
No transcript parser, new event store, history import/index or live benchmark
was created. Raw receipts remain private (directory 0700, files 0600).

The original retained bundle is NOT a Git repository. Its historical source
SHA is unknown, not the current consumer SHA. Seven source SHA-256 digests are
frozen in [historical120-audit.json](historical120-audit.json). No source client,
credentials, request IDs, private paths or transcript contents are published.
The separately preserved native-event digests bind the discovery chain:

| Native tool-output time (UTC) | Source-event SHA-256 | Observation |
|---|---|---|
| 2026-09-21T21:45:23.106Z | `18aab1bf7fd80a0fde6887627bc5adfad724c8fdca1d243f93fbe5363773e61d` | Groq and Cerebras catalog requests returned HTTP403, body code1010 |
| 2026-09-21T21:50:33.461Z | `b60afdc5a88e04290bca24539dde0a1c3bf85d096868d7f0a997f5cad6c9928e` | DSH full-turn Groq413, requested8738 against TPM8000; shell wrapper nevertheless printed exit0 |
| 2026-09-21T21:52:15.521Z | `f075b32d86a0ae173ca5fc606169a11cbcdbcf87099c2db1ff5c21c73e79be15` | Initial five-model benchmark output; Groq Qwen text429 |
| 2026-09-21T21:56:45.474Z | `4a9591e034f94f8ea0551305c87ad06ce8c3d6982c6bb2193438acd8ab03ffa2` | Re-derived summaries from120 retained receipts |

The 413 and truncated Qwen429 error outputs identify the same historical Groq
organization: scope SHA-256
`635b3d2fa5d15b2f488cdf6ae21d403f123972768e11301dfe2039224a914502`.
Individual successful rows do not contain account identity; their account scope
cannot independently be asserted. Cerebras account scope is unknown. The403
catalog failures are not proof of depleted credit or billing denial. The later
7/11 Qwen series and Cerebras402 lead remain outside this recovered cohort.

## Reconciled inventory

The benchmark contains120 logical rows:24 per provider/model (12 text,12 tool).
There are119 unique provider+model+response-ID identities, all chatcmpl-UUID,
plus one429 with no response ID. The source client makes one nonstreaming POST
per invocation and has no retry loop. This supports a120-attempt interpretation,
not a complete physical network audit: failed-request identity and transport
telemetry are missing. Never count the120 Tokenomics copies as120 extra calls.

| Observed provider/model (not corrections to the literal brief) | HTTP200 / rows | Tool-call presence /12 | Input tokens observed | Output tokens observed | Usage coverage |
|---|---:|---:|---:|---:|---:|
| groq/openai/gpt-oss-20b |24/24|12|2796|2091|24/24|
| groq/openai/gpt-oss-120b |24/24|12|2796|2233|24/24|
| groq/qwen/qwen3.8-27b |23/24|12|3889|1034|23/24|
| cerebras/gpt-oss-120b |24/24|12|2748|1812|24/24|
| cerebras/qwen-3.8-27b |24/24|12|4824|1512|24/24|

Observed input17053/output8682 excludes unknown usage for the429. Groq cache
fields are absent; Cerebras cache fields are present on48 rows (values0 or128).
Reasoning tokens are not retained; do not add an invented reasoning allowance.
No provider request timestamps survive the benchmark projection. Tokenomics
emission starts2026-09-21T21:55:05.135035Z, NOT a provider-request timestamp.

## RED claims and deterministic controls

1. **“12/12 verified tool outcomes” fails source inspection.** The benchmark
   marks success when a nonempty tool_calls list exists; it never executes the
   tool or validates its arguments. Text success means HTTP200; raw
   verified_outcome additionally requires nonempty text. Only116 raw rows carry
   that flag, versus119 HTTP200 rows, and neither is independent gold. Exported
   Tokenomics rows correctly have no outcome objects. Recovered natural task
   quality remains unknown; these are repeated synthetic smokes, not development
   tasks or120 independent work-item groups.
2. **“Measured free cost” fails provenance.** emit_tokenomics.py hardcodes
   cost_usd=0.0 on every event. No invoice/credit evidence backs it. All120 zeros
   are serialized assertions, not measured free entitlement. Parent, worker,
   verifier, retry/failure and invoice full-cost coverage is incomplete.
3. **“TTFT” fails instrumentation.** The client times urlopen plus full response
   body read; no first-token observation exists. Completion tokens divided by
   whole request duration is end-to-end output rate, not decode throughput.
4. **Apparent latency improvement is a denominator artifact.** Initial Groq
   Qwen text median463.8ms excludes its failed request. Existing derive_rows.py
   includes the151.1ms429 and reports385.3ms. Both were reproduced; the lower
   number is NOT a performance win. Same inputs, changed success conditioning.
5. **“Reconciled requests preserve identity” fails the historical mapping.**
   Tokenomics retains119 IDs in extra.request_id, but reconcile.py never flattens
   that field for Kerdoios UsageReceipt. The unchanged consumer gets zero request
   IDs and zero retry-state fields; its retried=0 is not observed zero retries.
   Its verified=0 reflects missing gold rather than120 failed tasks. A deliberate
   duplicated-input control counts48 requests per pair instead of24: this sum
   function expects curated input, it is not a physical deduplicator. The
   experiment uses the authentic119 IDs as the deterministic uniqueness control;
   the failed row remains unmatched by provider identity.
6. **Daily quota truth was not established by the old replay.** reconcile.py
   injects the same RPD14400/TPD200000 ceiling for every pair. No matching account
   receipt is supplied by that script. Its restart test is ledger plumbing, not
   proof of today's limits, balance, or Cerebras renewing free entitlement.

The strongest simple control is retaining existing request identities and
strictly separating presence, transport, outcome and economic coverage. A new
model cannot recover missing telemetry or turn these flags into independent
outcomes. No router/parser/quota database is needed or authorized here.

## Reproduction and verification

Read-only consumers used:
- Tokenomics `3fa33ee83d02e43a37d7890656a4151e200f2275`; fresh default
  `65e8f2dd9d6c5784a2a1c67f5236d69a5ae94021` differs only in zer0.repo.yaml.
  Thus the tested consumer code equals the fresh incumbent, not a stale rival.
- Kerdoios `9a74ed073c9a4721ae55e9cd7b058dabfe444ce4`; receipts.py is identical
  to fresh default `cdb43b05a328a50649d5120bffcdecea09492219`.

With those external checkouts on PYTHONPATH, bytecode disabled:

    python historical120_audit.py ORIGINAL_PRIVATE_BUNDLE

This bounded projection uses existing Tokenomics iter_jsonl and Kerdoios
UsageReceipt/summarize_receipts unchanged. It asserts source hashes, cardinality,
unique identity joins, status/usage/latency preservation, lost identity coverage,
and the deliberately duplicated-input control; emits sanitized aggregates only.
Original derive_rows.py was also executed unchanged with a private output cwd;
its output matches the retained bench2.json byte-for-byte. No live source script
was executed. Existing Tokenomics tests (trace_accounting, aggregate, adapters,
jsonl):24 passed, exit0. Kerdoios test_quota_*.py:44 passed, exit0. Initial narrow
Kerdoios test_quota_receipts.py selection matched no tests (exit5), retained as
an unsuccessful discovery command, not a pass.

These are frozen owning plumbing checks and an analytical replay, not an
independent efficacy assessment. No protected evaluator changed. No matched
frontier/current-provider arm ran. Backend revisions unknown despite common ID
shape; do not compare these timings with another date/backend as a code win.

## Next useful leaf and credit

Request exact-head independent review on EL23: verify the119-ID join, synthetic
outcome overclaim, hardcoded zero cost and Qwen latency denominator discrepancy.
Next bounded historical leaf: locate the distinct7/11 series and402 evidence,
then mine actual low-risk task groups rather than repeating this recovered120
smoke audit. Current free entitlement/governed admission remains a separate
Kerdoios dependency. Use the existing Shadow Network/Tokenomics identity seams
if a future authorized source repair is requested; no production change here.

Kevin/kvnloo supplied the historical objective and this mission. Historical DSH
worker authorship beyond the retained session is not independently resolved;
no claim that Hermes secondary authored its benchmark/client. Original
Tokenomics/Kerdoios contributors retain implementation and test credit. Hermes
secondary performed source recovery and this read-only audit. Reviewer: none.
Coordinator uses existing subscription; tokens/time are nonzero, invoice and
host joules unknown. Zero new Groq/Cerebras calls is not zero total compute.
