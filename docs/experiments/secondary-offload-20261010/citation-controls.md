# Cycle 7 — reuse path grounding, not a truth or promotion gate

PARTIAL development characterization; performance NOT_COMPARABLE;
independent_review=NOT_RUN. No model arm, live provider call, outcome promotion,
production change, protected evaluation or new parser/router/judge.

Requirement: [EL23 assignment](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094586102).
[Claim](https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6095117402).
Baseline `01b3b1459334b27e5e845210742f63af48462a26`.
Current claims/open-PR paths checked; no collision in this directory. PR35/36
had successful test checks and empty conversation/review/inline feedback.
Other failing PRs are not declared clean. This slice does not repeat the prior
historical cohorts, skill/EL31 studies or collector experiment.

## Existing implementation and revision correction

The existing `z0int.verification_density.grounding` is the smaller alternative
to writing a citation parser. Its documented scope is **grounding, not truth**.
It emits low-confidence soft signals in its shipped configuration. The existing
owning suite temporarily promotes some signals for logic tests; the probe leaves
`PROMOTED` empty. Neither positive nor negative soft signals become task gold.

Read-only retained-source discovery found this function, but that checkout's Git
link is invalid. Current default
`5873723eb33bd5c09e5220b3619e25e01efacd64` does NOT contain the module or its tests:
contents lookup returned404 and recursive tree lookup found no matching paths.
Thus the claim's initial default-head source lead was insufficient. Paginated
branch discovery recovered the actual existing feature head:

- [verification-density feature source](https://github.com/kvnloo/z0intelligence/blob/5a9ce7cf6e98b486668ad91b806a9261fdb4c7e4/src/z0int/verification_density.py)
- [unchanged owning tests](https://github.com/kvnloo/z0intelligence/blob/5a9ce7cf6e98b486668ad91b806a9261fdb4c7e4/tests/test_verification_density.py)

Exact remote source was exported privately; its module digest matches the retained
bytes. No current-default deployment is inferred. This is feature-source reuse
qualification, not a current production-incumbent A/B. Grounding module SHA256:
`8f1d10d6e53a6e7e0f51164daa48ae7ec919ad1cde80a5a61300afa25cf131b8`.
Owning test SHA256:
`2423650bacfee0aa9d9225a6514a04e5022d6d064b1d5c2eaba2cde6f5341221`.

## Natural source facts, synthetic controls explicitly separated

Reuse two previously frozen development groups, EL28 and z0evals81. EL36 is not
needed for this citation question. No new independent natural task was acquired.
The fifteen statements below are deliberate controls authored for this probe,
NOT observed provider outputs, real failure frequencies or held-out labels.

Source files:
- EL28 `.github/workflows/test.yml` at
  `26d92191440449eac7cd6cdd783d0d6c646e46a6`;
- the same file at current nightly
  `0212fee31a78cdba506f4d2684813962bfe0efb0`;
- z0evals81 `studies/aodl-admission-v1/golden-trace-manifest.yaml` at
  `6652713e25fb159a047457acfe348bd5ec98e146`.

Staged public source bytes are digest-bound and unchanged. Controlled file mtimes
100 and turn end200 are fixture values, not historical event times. The missing
base, edit marker and post-answer timestamp are synthetic controls too. No private
histories or hidden reasoning were read for this experiment.

| Control | Existing function result | Boundary demonstrated |
|---|---|---|
| Existing workflow path:13 | grounded, soft | Path/start line exists |
| False claim: old path:13 installs JAX extra | grounded, soft | No entailment check |
| Missing source path | ungrounded, soft | Missing-path signal works |
| Workflow:999 with pre-answer mtime | ungrounded, soft | Beyond-EOF signal works |
| Workflow:0 | grounded, soft | Not strict 1-based argument validation |
| Workflow:one-past-EOF | grounded, soft | Existing `n + 1` tolerance |
| Workflow:13-999 | grounded, soft | Range end is not checked |
| Old SHA stated, current source directory supplied | grounded, soft | Prose SHA is not a binding |
| Existing manifest:24 | grounded, soft | Path/start line exists |
| False claim: manifest:24 proves all real tasks succeed | grounded, soft | Source existence does not prove efficacy |
| URL citation | no signal | URLs are excluded, not accepted |
| Explicit abstention without citation | no signal | Cannot certify abstention correctness |
| Missing base directory | no signal | Unmeasurable, not a pass |
| Edited turn | no signal | Deliberately excluded by owner contract |
| Workflow:999 with post-answer mtime | grounded, soft | EOF check omitted when temporal evidence is uncertain |

These are characterization results, not an allegation that the owner promises a
strict citation security gate. Path/line0, range end, temporal uncertainty and
revision binding would matter if this helper were repurposed as one. No such
repurposing or production fix is authorized by this slice.

Meaningful RED: a test demanding that a grounded signal reject the deliberately
false JAX statement exits1. That invalid substitution remains RED; changing the
owner's intended semantics merely to get GREEN would be the wrong intervention.
The existing stronger no-model control is `git show EXACT_SHA:PATH` plus inspecting
the actual line, not a model explanation and not this soft signal:

- old line13 is `- run: pip install -e .`;
- current line15 is `- run: pip install -e ".[jax]"`.

The probe asserts those literal source facts. `git show` also rejects the missing
path. No new line parser or universal entailment predicate was implemented.
The prior fix remains credited to lesseradmin / Claude Opus5.5; no duplicate fix.

## Consequences for the two offload hypotheses

1. Evidence selection: retain exact-revision source retrieval as the baseline.
   Existing grounding may supply a soft diagnostic, never a whitelist or authority
   decision. On these examples no information gap requiring a model is shown.
2. Cited explanation: DISCARD the proposal to certify explanation truth using
   path grounding alone. False explanations receive the same signal as valid
   citations. An independently evaluated explanation arm is still unproven.

No-signal must stay unknown. Do not treat missing evidence, an edited turn or a
URL as success. Parent retains repair/merge/billing/retry authority; this probe
executes no text from citations. It does not establish path containment or safety
for arbitrary untrusted file reads. No routing integration should be added now.
All five literal Groq/Cerebras pairs remain NOT_ADMITTED with current entitlement
unknown; no account/doc refresh or provider invocation was needed in this slice.
Fresh frontier-parent and provider arms NOT_RUN. No response-ID/backend comparison,
TTFT, p50/p95, token savings or efficacy claims. Hosted energy and full
parent+worker+verifier accounting/invoice cost remain unknown.

## Reproduction, tests and limitations

[citation_probe.py](citation_probe.py) imports the unchanged implementation and
uses fixed controls only. It validates digests for the module, package, outcome
helper, manifest and both workflow versions before staging or importing. It does
not parse transcripts, create a store, mutate a judge or add a production API.

From this Evolution Lab checkout, with exact public source exports and a fresh
private output directory:

    PYTHONDONTWRITEBYTECODE=1 python3.12 docs/experiments/secondary-offload-20261010/citation_probe.py Z0_SOURCE ZE_SOURCE PRIVATE_OUTPUT
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Z0_SOURCE/src python3.12 -m pytest -q -p no:cacheprovider Z0_SOURCE/tests/test_verification_density.py

Executed results:
- owning focused grounding/default-confidence checks: 2 PASS,15 deselected;
- unchanged full verification-density suite:17 PASS on Python3.14;
- supported Python3.12 with manifest-pinned pytest8.4.2:17 PASS;
- fifteen fixed controls PASS; identical JSON on Python3.12 and3.14;
- altered module and altered manifest bytes each reject before staging, exit1;
- invalid truth-gate substitution exits1 as expected;
- first probe invocation failed on a relative import; corrected to import the
  existing package, not copy/rewrite owner code;
- Python3.12 initially lacked pytest; installed only in the private test venv;
- default Python gym-smoke failed before execution (`numpy` absent). This is an
  environment failure, not a successful runtime check; external exact-head CI is
  reported separately in the issue receipt.

Private result JSON SHA256:
`b22cb25eae08a2f312ba50fa4a5d3bb8b3e041fe29aef89d9fbcefa4f45778ee`.
Private files use restrictive creation permissions; published code/docs contain
only public source identities and sanitized derived observations.
Independent coordinator review requested at this artifact head: reproduce all
controls, source identities and default soft confidence; decide whether any strict
validator is actually needed before proposing an owner change. Sealed outcome
qualification NOT_RUN. Earlier reviews do not transfer to this head.

Credit: Kevin/kvnloo supplied the mission; existing z0intelligence implementation
and tests belong to their original contributors (feature head credits lesseradmin
and Claude Opus5.5). Hermes secondary recovered/reused/characterized them.

Next useful leaf: a bounded final reconciliation of remaining unrecovered failure
leads (especially the reported Qwen7/11 series) against the retained inventory;
only pursue an unexamined source, not another run of completed cohorts. If no new
source exists, consolidate exact-head coverage and independent-review gaps rather
than manufacture a live offload experiment. HOST owns the remaining cycle/deadline.
