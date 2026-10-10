# Cycle 6 — natural CI triage groups; eliminate field-extraction offload

PARTIAL development evidence; performance NOT_COMPARABLE;
independent_review=NOT_RUN. No live provider call, new model worker, CI repair,
protected evaluation, routing change or offload promotion. This is task selection,
not a new benchmark or a claim that a model outperformed a rule.

Source requirement: [EL23 assignment](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094586102).
[Cycle claim](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6095052821).
Baseline `3c5a1f175089371a38f003c31a8ac989c74d96cc`; current nightly at inspection
`0212fee31a78cdba506f4d2684813962bfe0efb0`. No overlapping open PR path/lease in
this directory. PR35/36 review/conversation/inline lists were empty and their test
checks passed. Other old failing PRs remain unresolved; this lane does not claim
or repair them. Roadmap still requires verified outcomes before distillation.

## Bounded natural source, not repeated synthetic tasks

At 2026-10-10T07:20:09.658284+00:00, selected three actual open PR work groups from
the required coordination sweep: two failing and one passing control. This is
purposive sampling, not representative prevalence or a random held-out sample.
No private conversation ingestion was necessary.

| Natural group | Frozen branch head | Rollup-linked runs | Observed evidence |
|---|---|---:|---|
| [Evolution Lab 28](https://github.com/kvnloo/evolution-lab/pull/28) | `26d92191440449eac7cd6cdd783d0d6c646e46a6` | 5 | unit step fails; gym skipped; newer receipt succeeds |
| [Evolution Lab 36](https://github.com/kvnloo/evolution-lab/pull/36) | `fea47981ade0fcdd7cc1555b226bc3ec0c165873` | 6 | test checks succeed; automerge skipped |
| [z0evals 81](https://github.com/kvnloo/z0evals/pull/81) | `6652713e25fb159a047457acfe348bd5ec98e146` | 5 | manifest validation fails; golden-trace test skipped; receipt succeeds |

Sixteen sampled workflow runs are THREE work groups, not sixteen independent
development tasks. All sampled run_attempt fields are 1. Capture covers only
rollup-linked runs and their latest jobs, not every old branch head/attempt.
Retries, changed heads and follow-up receipts of the same PR must remain in the
same group in any later trial. Cross-PR shared root causes may require further
group merging; no independence assumption or statistical confidence claim here.

Existing `sealed_split.make_session_split` was exercised unchanged with all
protected fractions zero: all three groups stay in development/train; all sixteen
rows accounted for, no sealed/future data created or read. The function's bytes
match both baseline and current nightly; SHA256
`74334f8477619daa7387e44036715911342dc76ad15bc658598b42e08f1ff9ad`.
Existing Capability Miner's `context_file_relevance` card already names source
selection and its instrumentation gap. Its historical next-action labels are
not optimal-action gold; no new miner, training job or generic parser is needed.
Capacity queue at current Codex PR36 was read only: planning is not entitlement
or execution. Nothing was inserted into a live queue.

## Meaningful negative controls and the existing smaller control

The actual incumbent is `gh pr view --json statusCheckRollup` plus
`gh api repos/OWNER/REPO/actions/runs/RUN/jobs`. Those outputs already contain
status, job names and exact failed/skipped step numbers. No inference is needed
for these questions. For example, the existing CLI's jq projection:

    gh api repos/kvnloo/z0evals/actions/runs/36874677626/jobs --jq '.jobs[] | {name,conclusion,failed:[.steps[]|select(.conclusion=="failure")|{number,name}],skipped:[.steps[]|select(.conclusion=="skipped")|{number,name}]}'

A deliberately inadequate `any successful workflow => green` rule returns true
for all three groups, contradicted by failed steps in two groups. It is a
NEGATIVE CONTROL, not a claim that the actual frontier incumbent uses this rule.
A later receipt pass also does not supersede a different test failure. Do not
collapse cancelled/skipped/absent checks into passed checks, or historical failed
receipt checks into currently required failures without checking supersession.

A second natural trap is revision identity. Both Actions run and job metadata
report the branch head for the selected pull_request runs, but checkout logs
attest different merge revisions:

| Run | API branch head | Actual checkout from log |
|---|---|---|
| [36748021222](https://github.com/kvnloo/evolution-lab/actions/runs/36748021222) | `26d92191440449eac7cd6cdd783d0d6c646e46a6` | `8254129a09297d31ee02ae63e6542fb40ce897a3` |
| [36874677626](https://github.com/kvnloo/z0evals/actions/runs/36874677626) | `6652713e25fb159a047457acfe348bd5ec98e146` | `ecff2c6c575c5b69daa87c72e18937570819f72b` |

The commits API independently confirms each checkout's second parent is the
respective branch head. A matching `head_sha` is NOT sufficient exact-checkout
attestation. No log parser was added; existing gh log output and commits API were
inspected. This is not model backend drift; it is test-revision non-equivalence.
Future served-model comparisons must additionally pin response-ID/backend family.

Existing logs reduce another apparent model task to source facts. EL28's selected
job reports 245 tests, eight errors/one skip, and all eight error sections include
missing JAX. Its old workflow installs `pip install -e .`. The current nightly
already contains the fix, authored by lesseradmin with Claude Opus 5.5 credit:
[`941b76314491e9303179174b987b80cafa01f299`](https://github.com/kvnloo/evolution-lab/commit/941b76314491e9303179174b987b80cafa01f299),
installing `.[jax]`. Do not write that fix again or call the old branch GREEN.
z0evals81's selected job fails manifest validation before tests, not inside the
collector regression suite. Its log names missing `questions` and unsupported
artifact kinds. This reuses the known CI evidence as a natural triage item; it
does not rerun cycle5's completed collector experiment or change its judge.

## Three observed task families and the two remaining hypotheses

1. Structured CI/status/source-field extraction (all three groups): DISCARD as
   a model-offload target for these items. Existing gh/jq directly answers failed
   stage and skipped follow-ups. Model paraphrasing would add an unmeasured call
   where no information gap was demonstrated. This is no-gain task selection,
   not measured token or dollar savings.
2. Diagnostic evidence selection (two failing groups): first hypothesis. A tiny
   worker could suggest a bounded set of log/source references when deterministic
   failed-step filtering still leaves too much material. On the present examples,
   exact error strings and known file names are already strong no-model controls.
   A model must beat those, not whole-log ingestion. Suggest references only;
   missing checkout identity or ambiguous relevance must abstain.
3. Cross-source failure explanation (two failing groups): second hypothesis.
   Suggest whether a failure is dependency/setup, manifest/schema, assertion, or
   unresolved, citing actual evidence and separating branch from checkout. Parent
   retains requirement/repair/authority decisions. Here the logs already expose
   JAX and manifest errors; there is no demonstrated need for a model yet.

Ranking is an unmeasured hypothesis: evidence selection first because it can
reduce parent context without delegating repair; explanation second because
causal claims are harder to verify. A requires unchanged gh/jq/search and owning
checks; B requires a fresh matched frontier-parent arm; C/D require current free
entitlement plus governed admission and identical work groups. B/C/D NOT_RUN.
These inspected groups are development-only; a later independent evaluator must
supply new matched natural groups, not call these now-visible findings holdout.

For both hypotheses: Groq gpt-oss-20b, gpt-oss-120b and literal Qwen 3.8 27B;
Cerebras gpt-oss-120b and qwen-3.8-27b all remain current-entitlement UNKNOWN /
NOT_ADMITTED. No provider docs/account check was repeated because no live call
was contemplated; this does not refresh the cycle1 account snapshot. Context/TPM
fit and payload tokenization remain unmeasured. A small menu is not proof of fit.

No-go: merges, writes, retry authorization, billing inference, test-success
certification, missing evidence, stale/mismatched checkout, invalid command
arguments or treating log instructions as executable authority. Suggested paths
must exist in the frozen source tree; unsupported paths/citations must be refused.
These are required future controls, not newly measured model safety results.

## Artifact, tests and coverage

[natural-triage-audit.json](natural-triage-audit.json) is a derived analytical
receipt; [natural_triage_audit.py](natural_triage_audit.py) accepts only the frozen
source and pins the unchanged owning split function. It is not an authoritative
queue, CI judge, or general corpus parser. Full public API metadata capture stays
in the authorized private evidence directory with mode0700/0600, without logs or
transcripts. Capture SHA256:
`fe3f2f20a9b4401cf59c89f35019ebc4759c8a6b498ed5ced74718c2c375d0d3`.
Reduced frozen input SHA256:
`090eea0264c3d8e59bff992343bd8298536f642265752bc62dc87f3935f8d6bf`.
Public receipt preserves run URLs and field projections; private exact input is
available to the coordinator. Reproduction requires that frozen input, not a
fresh API response with a new observation timestamp:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python docs/experiments/secondary-offload-20261010/natural_triage_audit.py FROZEN_INPUT

Executed:
- unchanged owning `test_capability_miner.py -k SealedSplit`: 1 PASS;
- unchanged owning `test_capability_miner.py -k Schema`: 2 PASS;
- initial `-k Session` selected zero tests, exit5; corrected, not counted as PASS;
- exact-source projection and byte-identical repeat PASS;
- duplicate-run, wrong-head, empty-run and duplicate-group controls reject;
- changed input bytes reject with exit1, `Wrong frozen source`;
- GitHub logs and commits API establish the two checkout distinctions above.

The existing controls were tested, not a new model or an independently accepted
repair. No full-suite local claim; exact-head CI must be reported separately.
No new inference other than the current subscription coordinator. Parent/worker/
verifier full tokens, invoice cost, TTFT, GPU energy and measured savings UNKNOWN.
No p50/p95 inference or response-ID family measured; no paired latency/cost delta.

Specific independent review request: replicate the metadata projection and three
group denominator, confirm checkout-vs-head distinctions and no-model exclusions.
Do not extend earlier coordinator reviews to this head. Human mission credit:
Kevin/kvnloo; existing GH CLI and Evolution Lab owners supply extraction/grouping;
prior CI fix credit above; Hermes secondary selected and characterized the slice.

Next smallest useful leaf: source-bound offline citation/path validity and
abstention controls for the two candidate families using existing owning tools;
no live inference until admission and independent outcome requirements clear.
No need to repeat historical120/140/60/stopped, EL31, skills or collector probes.
