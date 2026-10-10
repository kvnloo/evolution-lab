# Cycle 5 — trace completeness is not outcome acceptance

Status: PARTIAL evidence / HOLD for offload promotion. Performance is
NOT_COMPARABLE; independent_review=NOT_RUN at this head. No provider call,
model subprocess, new router, production change, evaluator edit, protected-gold
access, or new natural-task efficacy result. This closes only the cycle-5 leaf.

Requirement: [EL23 assignment](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094586102)
requires independently verified task outcomes, not transport/collector success.
[z0evals#58](https://github.com/kvnloo/z0evals/issues/58) links the existing golden
collector [PR81](https://github.com/kvnloo/z0evals/pull/81). Reuse that collector;
do not write another judge. Claim: [6094977862](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094977862).
Evolution Lab baseline: `67bcbd9b1f889c6a8810b874af5547ce140263c8`.

## Exact sources and scope

- Collector, importer, public UNIT fixture/tests and schemas: z0evals
  [`6652713e25fb159a047457acfe348bd5ec98e146`](https://github.com/kvnloo/z0evals/tree/6652713e25fb159a047457acfe348bd5ec98e146/studies/aodl-admission-v1),
  still PR81's current head at inspection; collector is absent on default main.
- Current default z0intelligence:
  [`5873723eb33bd5c09e5220b3619e25e01efacd64`](https://github.com/kvnloo/z0intelligence/tree/5873723eb33bd5c09e5220b3619e25e01efacd64/src/z0int).
  Used unchanged `receipt.Outcome`, `hermes_decisions.observed`,
  `decision_opportunity.build_decision_opportunity` and `deterministic_gate`.
- Collector manifest pins a DIFFERENT implementation revision,
  `28db1eecbe9fa9921c52945e15d4894b6dddae3d`. This exercise is a cross-contract
  UNIT characterization, not the manifest-qualified live canary.
- The retained local z0int snapshot had an invalid Git link. Its Hermes decision,
  DecisionOpportunity and Tokenomics emitter bytes differ from current default;
  its receipt.py bytes match current default. Used exact remote source exports,
  not an invented revision for that local tree. No peer checkout repaired/edited.
- [trace_seam_probe.py](trace_seam_probe.py) pins hashes of all imported owning
  source files, schemas and fixture code. [trace-seam.json](trace-seam.json) is
  actual offline output. External code is imported unchanged, not vendored.

## Baseline failure and simplest existing control

The collector's `verified_outcome()` accepts a row with `outcome_tier=gold` when
any recognized outcome field is non-null. The CLI's `--require-verified` then
checks completeness, not positive outcome semantics. A downstream consumer
using exit 0 alone as verified success would trust inconsistent tier metadata.

Using ONLY the existing public `base_rows()` unit fixture, the unchanged CLI gave:

| Deliberately supplied outcome | Tier supplied | Structural complete | Outcome complete | CLI exit |
|---|---|---|---|---|
| absent | absent | true | false | 1 |
| verified_success=true | gold | true | true | 0 |
| verified_success=false | gold (inconsistent) | true | true | 0 |
| verified_success=true, test_pass=false | gold (inconsistent) | true | true | 0 |

The unchanged importer's `validate_bundle()` also accepts the public test bundle
with its nested outcome changed to verified_success=false while its completeness
flags stay true. Validation only was called; no import into evaluator results.
This is an adversarial metadata-consistency control, NOT evidence that the
canonical producer emits these contradictory rows, or that any real task failed.
The fixture's declared provider/execution labels are synthetic; zero requests
were sent and no provider/model latency or response-ID family was measured.

The existing owner already has the needed semantics: `receipt.Outcome.tier()`
classifies both negative and contradictory cases as `negative`, and
`is_verified()` returns false. Feeding those owner-derived tiers to the same
collector makes outcome_complete=false; positive-only remains true. This is the
matched no-model control. No new success predicate, judge or routing code needed.
It does NOT authorize changing the frozen collector here. An owning review must
decide whether importer hardening or a documented trusted-producer boundary is
wanted. Preserve negative outcome rows and their cost even when not promoted.

Existing protections also worked: missing output usage and a wrong dispatch
permit join both fail structural completion. These controls prevent one observed
consistency gap from being inflated into a claim that the whole collector fails.

## Existing Shadow Network boundary

Current `hermes_decisions.observed()` returns `answered` for a completed turn
even when a supplied verified_success flag is false. That function describes
behaviour, not quality, as its documentation says. Do not reinterpret an
ACT->answered count as verified development work. DecisionOpportunity leaves
`expected_outcome.verifier` null; trace/semantic identity is not an independent
verifier. Existing deterministic gate returns ASK for missing required branch
facts and for an unauthorized write. Neither case needs a small-model opinion.

Keep the following existing identities distinct, without a new store:

| Existing seam | Evidence it provides | What it does not prove |
|---|---|---|
| DecisionOpportunity semantic_id / trace / opportunity_id | pinned intent/evidence and observation identity | task outcome or provider request identity |
| dispatch caller_trace_id / admission receipt / permit_dispatch_id | structural authority linkage | current free credit or all-attempt economics |
| physical authority_dispatch_id / usage | matched execution and known input/output fields | cache/reasoning/parent/verifier cost coverage |
| Tokenomics emit_provider_usage logical_source_id / physical_source_id, role, attribution | existing identity/accounting vocabulary | unknown invoice, automatic gold, full request deduplication |
| owner Outcome tier plus independent verification source | explicit positive/negative/execution distinction | validity of an untrusted tier label or model self-grade |

Smallest useful integration is therefore an evidence handoff through these
existing joins, with owner-normalized outcomes and unchanged independent
verification, not a new Shadow Network service. Keep failed/unknown attempts in
the cost denominator. Native provider response ID is still separate from the
orchestrator's trace, and drift/mixed backend families remain NOT_COMPARABLE.

## Verification and review queue

Executed unchanged in isolated private output directories:

- z0evals `pytest tests/test_golden_trace.py`: 11 passed, exit 0.
- Current z0int `test_decision_opportunity.py -k 'not real_packet_builder'`:
  11 passed, 1 deselected, exit 0. Full packet-building integration NOT_RUN.
- Current z0int `test_hermes_decisions.py -k 'non_user_traffic or decisions_report_joins'`:
  12 passed, 11 deselected, exit 0. Plugin/hooks/CLI integration NOT_RUN.
- Probe: actual four CLI invocations with exits above, owner normalization,
  importer validation, missing-usage/wrong-link and missing-fact/authority checks
  passed; all pinned imported source hashes unchanged afterwards.
- Default Python lacked jsonschema; an initially tried shared Python path did
  not exist. Used a new private test-only environment (Python 3.14.7,
  pytest 9.1.1, jsonschema 4.26.0). No peer/shared environment changed.

Reproduce with the exact external source exports at SOURCE_DIRECTORY/z0evals
and SOURCE_DIRECTORY/z0intelligence, and a new empty private output path:

    PYTHONDONTWRITEBYTECODE=1 python trace_seam_probe.py SOURCE_DIRECTORY PRIVATE_OUTPUT_DIRECTORY

PR81's existing CI is NOT GREEN: run
[36874677626](https://github.com/kvnloo/z0evals/actions/runs/36874677626) failed
on its merge revision `ecff2c6c575c5b69daa87c72e18937570819f72b` before tests,
with four manifest validation errors (missing questions; unsupported artifact
kinds frozen-cohort, result-schema, runner). This is not an exact-PR-head test
failure or evidence that the local focused tests failed. No inline review items
were present. The manifest/CI repair is outside this lane's write scope.
Evolution Lab PR35/36 had passing test checks and no review/comments; all open
PR file lists were inspected for collision with this owned directory.

Independent review request: reproduce the inconsistent-tier acceptance and the
unchanged owner-normalized rejection at these exact revisions; distinguish a
trusted-input contract from a promotion-safety gap. No protected judge change is
requested from this worker. Coordinator cycle-3 analytical review
[6094962666](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094962666)
applies only to `7acf8b18d1414bfb406cbdaccafe4daf86558843`, not this new head.
Later coordinator cycle-4 review
[6095010999](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6095010999)
also reports narrow analytical PASS at this cycle's baseline
`67bcbd9b1f889c6a8810b874af5547ce140263c8`; it does not review this new probe.

Costs: zero new Groq/Cerebras calls; current entitlement/admission still unproven.
Coordinator subscription usage is nonzero and running-session accounting is
incomplete; parent/worker/verifier invoice cost, GPU energy and savings UNKNOWN.
No current frontier arm, natural matched groups or provider arm were run.

Credit: Kevin/kvnloo supplied the mission; original z0intelligence/z0evals authors
own their code and public tests; Hermes secondary characterized this seam. No
claim to an independently accepted development outcome. Next independent leaf:
freeze 2–4 recurring natural low-risk work families from bounded existing receipts,
keep branch/retry descendants together, and compare source-field extraction with
model-needed cases. Do not repeat completed historical or skill/EL31 studies.
