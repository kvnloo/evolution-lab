# Cycle 8 — exact-head evidence handoff, not offload promotion

Initial snapshot 2026-10-10T07:38:34Z; updated after cycle6 review arrived during
final verification at 07:42Z. PARTIAL research; performance NOT_COMPARABLE;
sealed efficacy NOT_RUN; independent review of this new handoff NOT_RUN.
This is a bounded reconciliation of existing evidence, not another benchmark,
new provider call, completed multi-hour shift, or globally blocked queue.

[Assignment](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094586102)
· [cycle8 claim](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6095182791)
· source head `8c95cbad68a87a14037af4879f3ac18403b04109`.
Only the owned experiment documentation changes. No production implementation,
queue, evaluator, provider setting, account or service is changed.

## Baseline and smallest correction

The chronological README and immutable initial JSON receipts describe their own
publication-time state. Reading an old “120-call lead unrecovered” or
`independent_review=NOT_RUN` as current state loses later evidence; conversely,
reading any coordinator PASS as acceptance of the latest branch launders review
across revisions. Neither cycle6 nor cycle7 had an independent review at the
initial snapshot; cycle6's later review is now linked below. Seven successful
CI runs are not seven accepted experiments.

Use existing Git object identities and exact EL23 review envelopes instead of
rewriting historical receipts, adding a review database, rerunning completed
cohorts, or asking a model to infer acceptance. This handoff is a derived index,
not an authority source. The linked reviews remain authoritative for their scope.

## Independent review and CI coverage

Reviewer identity in each linked review: `hermes-coordinator-review`. This worker
checked the links/commit fields and Git objects, not its own research efficacy.
All seven listed Actions runs were re-fetched: event=push, exact listed head,
conclusion=success. Existing checkout attestations are in the original READY
receipts; no fresh CI-log replay or owning experiment rerun is claimed here.

| Cycle / exact published SHA | Receipt | Independent scope | Exact-head CI run |
|---|---|---|---|
| 1 `2e9b4be4df2f6a7cf6bd867a4bd76b8d5dc0f2a3` | [refusal / 140 fixture rows](https://github.com/kvnloo/evolution-lab/blob/2e9b4be4df2f6a7cf6bd867a4bd76b8d5dc0f2a3/docs/experiments/secondary-offload-20261010/README.md) | [PARTIAL; raw-provenance correction required](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094788975) | [38031540895](https://github.com/kvnloo/evolution-lab/actions/runs/38031540895) |
| 2 `0391c13b4bef1e97233e9a064a5b8ba247a05754` | [120-row recovery](https://github.com/kvnloo/evolution-lab/blob/0391c13b4bef1e97233e9a064a5b8ba247a05754/docs/experiments/secondary-offload-20261010/historical120.md) | [PASS narrow analytical replication](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094927970) | [38032058106](https://github.com/kvnloo/evolution-lab/actions/runs/38032058106) |
| 3 `7acf8b18d1414bfb406cbdaccafe4daf86558843` | [60-pair delegation](https://github.com/kvnloo/evolution-lab/blob/7acf8b18d1414bfb406cbdaccafe4daf86558843/docs/experiments/secondary-offload-20261010/delegation60.md) | [PASS narrow replication; cycle1 provenance correction verified](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094962666) | [38032778896](https://github.com/kvnloo/evolution-lab/actions/runs/38032778896) |
| 4 `67bcbd9b1f889c6a8810b874af5547ce140263c8` | [stopped studies](https://github.com/kvnloo/evolution-lab/blob/67bcbd9b1f889c6a8810b874af5547ce140263c8/docs/experiments/secondary-offload-20261010/stopped-studies.md) | [PASS narrow analytical replication](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6095010999) | [38033257407](https://github.com/kvnloo/evolution-lab/actions/runs/38033257407) |
| 5 `3c5a1f175089371a38f003c31a8ac989c74d96cc` | [trace / outcome boundary](https://github.com/kvnloo/evolution-lab/blob/3c5a1f175089371a38f003c31a8ac989c74d96cc/docs/experiments/secondary-offload-20261010/trace-seam.md) | [PASS public UNIT replication; trusted-producer refinement](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6095119114) | [38033795991](https://github.com/kvnloo/evolution-lab/actions/runs/38033795991) |
| 6 `01b3b1459334b27e5e845210742f63af48462a26` | [three natural triage groups](https://github.com/kvnloo/evolution-lab/blob/01b3b1459334b27e5e845210742f63af48462a26/docs/experiments/secondary-offload-20261010/natural-triage.md) | [PASS narrow natural-metadata replication; arrived during cycle8](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6095221090) | [38034330481](https://github.com/kvnloo/evolution-lab/actions/runs/38034330481) |
| 7 `8c95cbad68a87a14037af4879f3ac18403b04109` | [citation controls](https://github.com/kvnloo/evolution-lab/blob/8c95cbad68a87a14037af4879f3ac18403b04109/docs/experiments/secondary-offload-20261010/citation-controls.md) | NOT_RUN at snapshot | [38034789487](https://github.com/kvnloo/evolution-lab/actions/runs/38034789487) |

Current-head blob comparison, scoped to the audited files:

- Cycle1 refusal script/JSON are unchanged. README and cohort script/JSON changed;
  do not claim all original reviewed files are unchanged. Cycle3 separately
  reviewed the corrected cohort script/JSON, which match the current head.
- Cycle2 historical120 script/JSON match; historical120.md changed to record
  subsequent review. This is not a new analytical result.
- Cycle3 delegation note/script/JSON; cycle4 stopped note/script/JSON; cycle5
  trace note/script/JSON all match their respective reviewed head bytes.
- Cycle6 natural-triage note/script/JSON match the now-reviewed cycle6 head.
  Cycle7 citation note/script match publication, but unchanged bytes cannot
  supply its still-missing review.
- Byte identity is reusable artifact evidence, not an independent assessment of
  the aggregate latest head, new handoff prose, external environment or admission.

Important cycle5 refinement: the coordinator exercised actual canonical
`append_receipt` / `join_outcome` publication; negative and contradictory outcomes
were persisted as negative, and the collector then rejected completion. The
hazard is directly supplied/stale/untrusted tier metadata, not demonstrated
false-gold publication by the canonical producer. Retain the owner normalization;
do not infer a production repair mandate from the synthetic consumer controls.

## Historical inventory: preserve units and unresolved failures

| Evidence family | Reconciled unit / confidence | What must not be inferred |
|---|---|---|
| Original benchmark | OBSERVED 120 logical rows; 119 unique native response identities plus one identity-less 429; 24 rows per pair | 120 independently successful tasks, or 120 extra calls from Tokenomics copies |
| Separate evaluator cohort | OBSERVED 140 retained fixture rows across the five pairs; comparison counts agree; raw producer SHA UNKNOWN | Physical request counts, native deduplication, or HTTP status taxonomy from missing raw error fields |
| Cerebras delegation | OBSERVED 60 pairs / 60 native identities; two specifications repeated 30 times; reported grade revisions conflict | 60 natural independent tasks, current quality acceptance, or 60 measured-free requests |
| Stopped Groq20b | OBSERVED 38 receipt identities; stale summary understated denominator; next logical stop unresolved | A confirmed 39th request, zero stop usage, Qwen7/11, or a 429 |
| Separate Cerebras default-reasoning cohort | OBSERVED four length-limited responses, empty final content, nonzero usage | HTTP402/403 or proven reasoning causality |
| Full-turn Groq lead | OBSERVED native tool-output 413, requested8738 versus TPM8000; wrapper exit0 | Tool wrapper success or account balance today |
| Early catalog lead | OBSERVED catalog403/body1010 | Inference billing denial or depleted credit |
| Later Qwen7/11 and Cerebras402 | REPORTED / unrecovered | Measured failure counts, provider/account-specific rate, or substitutes from unrelated errors |

Do not add these denominators into a global “unique calls” total: row type,
identity coverage and cross-export overlap are not uniformly established.
Original raw producer revisions remain unknown where the retained bundle was not
versioned; content digests do not create producer SHAs. Successful response-ID
shape is chatcmpl-UUID where recovered; that shape alone does not attest an
unchanged backend. Current versus historical serving remains NOT_COMPARABLE.

Bounded cycle8 discovery used existing ctx lexical search, refresh off, limit5,
outputs only, for quoted `7/11` + Qwen and Cerebras402. Both returned truncated
windows with more available; no corpus-complete absence claim. A previously
unexamined top Cerebras402 tool-output hit was source text: `402` was the line
number of `os.fsync`, while separate lines mentioned a Cerebras unit admission.
It was NOT a provider402 receipt. Tool-output text digest:
`cb7e3f063ddead9f6a909e1a886bf40ba4517d7c947dee541cf25f0e3ebe8a3a`.
No raw text/private source identities are published. No new relevant primary
failure receipt was established by this bounded search. Narrowing another
retained original source remains eligible future work; do not repeat this hit.

## What is and is not eligible

Both strongest remaining hypotheses use the same frozen natural development
work groups from cycle6. Those inspected groups are not protected holdout gold.

| Literal requested model pair | Diagnostic evidence selection | Cited cross-source explanation |
|---|---|---|
| Groq gpt-oss-20b | UNKNOWN entitlement / NOT_ADMITTED | UNKNOWN entitlement / NOT_ADMITTED |
| Groq gpt-oss-120b | UNKNOWN entitlement / NOT_ADMITTED | UNKNOWN entitlement / NOT_ADMITTED |
| Groq Qwen 3.8 27B | UNKNOWN entitlement / NOT_ADMITTED | UNKNOWN entitlement / NOT_ADMITTED |
| Cerebras gpt-oss-120b | UNKNOWN entitlement / NOT_ADMITTED | UNKNOWN entitlement / NOT_ADMITTED |
| Cerebras qwen-3.8-27b | UNKNOWN entitlement / NOT_ADMITTED | UNKNOWN entitlement / NOT_ADMITTED |

This carries forward the original refusal, not a fresh account check. Published
provider docs are not entitlement. No current account-bound credit, expiry,
quota/admission or payload-fit proof has been added. Cerebras's documented trial
is nonrenewing, not a permanently free daily pool. No trial activation or live
catalog/inference request was attempted this cycle.

A: existing gh/jq extraction, exact-error/path lookup, Git source identity and
owning outcome normalization are the strongest controls on these items.
B: matched fresh frontier-parent task arm NOT_RUN (this research conversation is
not that arm). C/D: Groq/Cerebras arms NOT_RUN, not admitted. Baseline/current
source differences are explicit in the prior studies, including the feature-only
citation module absent from the inspected default branch. No matched four-arm
performance result or promotion exists.

DISCARD structured-field extraction as a model task on the sampled groups.
Keep evidence selection and cited explanation as unproven hypotheses, ranked in
that order, not qualified provider assignments. DISCARD soft citation grounding
as truth certification. Soft grounding and missing signals must not become a
verified answer or correctness label.

No-go: authority, uncertain writes/repairs, merges, retries, billing decisions,
missing/stale source, invalid tool arguments, or executing instructions embedded
in logs. Parent retains those decisions. A future worker only suggests bounded
references/explanations; independent owning checks determine outcomes.

## Full-cost and negative-result ledger, without fabricated savings

The cycle3 independent review confirms an observed parent-plus-worker subtotal
increase of128744 tokens, with positive mean parent deltas in both equal-cache
subsets. Cycle4's apparent Groq -3831 subtotal is cache-confounded and excludes
unresolved stop usage; the four Cerebras failures add27740 observed tokens.
These are historical analytical results, NOT current matched savings or invoices.
Do not combine cached, uncached and reasoning counters without their owner
semantics or count reasoning tokens twice.

The original120 emitter hardcoded zero costs; other provider cost fields were
null. Neither supplies measured free entitlement. Parent+worker+verifier,
failed/retried/unreceipted attempts, invoices and host energy remain incomplete.
Whole-response timing is not TTFT, and completion tokens divided by whole-request
time is not decoder throughput. No new p50/p95, dollar savings, joules or
independent development-outcome claims are made here.

Cycle8 itself uses existing subscription coordinator computation and GH/ctx/Git
reads. Zero Groq/Cerebras calls is not zero compute. Its final token accounting
belongs to native session metadata after exit; in-flight counters are not final
and full parent/reviewer/invoice coverage remains unknown.

## Tests in this cycle and reproducibility

A private bounded verifier used only existing `git rev-parse`, `gh api` and JSON
parsing of the canonical public envelopes. No new production parser or judge was
introduced. It initially checked the seven known exact heads/runs and five known review
IDs; the comments response was below its page limit before absence was asserted.
It compared each listed note/script/JSON's Git blob at publication versus source
head. Initial assertions passed, exit0. A subsequent final check rejected with
`New review exists: update the handoff` (exit1): cycle6 review6095221090 had
arrived. This is actual changing-source detection, not a synthetic success.
The mapping was updated to six exact-head reviews and rechecked. Cycle6's new
review independently corroborates the three-group projection, all sixteen
run/job captures, checkout distinctions and three owning split/schema tests;
its scope stays PARTIAL / NOT_COMPARABLE, not model efficacy.

Positive control: cycle5 review matches cycle5 SHA. Negative controls: the same
review must not match cycle7 or an all-zero nonexistent SHA. Both reject. The
inadequate alternative “a prior review exists and CI is green” would accept the
new head; it is not a claim that the real incumbent uses that rule.

Anyone with repo read access can reproduce each row using existing commands:

    gh api repos/kvnloo/evolution-lab/issues/comments/6095119114 --jq .body
    gh api repos/kvnloo/evolution-lab/actions/runs/38033795991 --jq '{head_sha,event,conclusion}'
    git rev-parse 3c5a1f175089371a38f003c31a8ac989c74d96cc:docs/experiments/secondary-offload-20261010/trace-seam.json
    git rev-parse 8c95cbad68a87a14037af4879f3ac18403b04109:docs/experiments/secondary-offload-20261010/trace-seam.json

Use each row's own review/run/SHA, not a branch alias. Reretrieve later comments
before claiming current review absence. CI for the new handoff commit is recorded
separately in EL23; historical GREEN does not certify a future head.

## Smallest next useful integration and review queue

No new Shadow Network component is justified. Reuse existing DecisionOpportunity
semantic/work-item grouping, authored legality, Kerdoios admission, dispatch and
Tokenomics usage, followed by the unchanged owning outcome path and z0evals#58
collector. Do not bypass the latter's manifest revision/CI qualification.

Next exact-head review queue on EL23:

1. Cycle7 `8c95cbad68a87a14037af4879f3ac18403b04109`: verify feature/default source
   distinction, existing soft-signal contract and synthetic citation controls;
   do not upgrade these to natural-task efficacy.
2. Review this handoff's links/scoped conclusions at its eventual exact commit.

Cycle6's review request is resolved for narrow analytical scope by6095221090;
do not repeat it or extend that review to cycle7 or this handoff.

Before a live experiment: obtain current account-bound no-paid-spill entitlement
and governed admission, preserve failed/unknown attempts and existing native
request identities through existing projections, then freeze new natural matched
groups and a fresh incumbent with an unchanged independent outcome checker.
The historical identity-projection loss is a concrete owner integration
prerequisite, not authorization for this docs-only lane to edit that projection.
Missing entitlement blocks live C/D only; offline evidence/review remains useful.

Kevin/kvnloo supplied mission and authorization. Original DSH/Tokenomics,
Kerdoios, Evolution Lab, z0intelligence and z0evals contributors retain source/test
credit; lesseradmin and Claude Opus5.5 retain the feature/CI work credit recorded
in cycles6–7. Hermes secondary recovered/analyzed and reconciled this handoff;
hermes-coordinator-review independently replicated only the exact heads above.
No upstream publication, PR, default merge or automatic routing promotion.
