# Factorized z0 stack v0

Tracking: https://github.com/kvnloo/evolution-lab/issues/27

## Question

Which parts of the z0 stack actually improve verified outcomes or systems cost,
and which are redundant once another mechanism is present?

The factors are deliberately separated by role:

```
raw observation
   |
   +-- SoL-Pi ObservationPack        observe / compress
   |
addressable evidence
   |
   +-- OMP RLM                       address / query
   |
current evidence
   |
   +-- StatePacket + unified memory  state / reconcile provenance
   |
legal bounded options
   |
   +-- local SLM                     decide / abstain
   |
proposed transition
   |
   +-- Bend                          verify structural invariants
```

This is not a new runtime stack. Evolution Lab only defines the experiment,
grouping, receipts, and promotion evidence.

## Phase 0

Run exactly six arms on the same grouped work-item cohort:

1. native control
2. SoL-Pi only
3. RLM only
4. StatePacket memory only
5. local SLM only
6. Bend only

Do not run a five-factor arm yet.

## Current concrete seams

### SoL-Pi

Use `kvnloo/sol-pi-hermes#2` first: frozen Hermes `CaptureResult`
observations -> ObservationPack shadow projection. Preserve the original
provider-visible observation as control. Measure retained evidence, bytes/tokens,
latency, and recall use.

Do not combine Action Fusion, OCC, or EPR in that first observation experiment.

### RLM

The richest downstream OMP experiment is currently
`kvnloo/oh-my-pi:exp/rlm-evidence-ab-clean`: it contains the context engine,
bounded evidence handles, depth-1 experiments, and explicit evidence A/B
orchestration/reporting. It is substantially behind current OMP main, so treat
its receipts as experimental evidence and rebase/reduce before any current-main
runtime dogfood.

The first comparison here is **RLM evidence addressing only**, not recursion as
a goal.

### Unified memory / StatePacket

Use z0intelligence #22/#63. Events/raw sources remain truth. Temporal history,
FTS/AgentsView, TencentDB, and StatePacket are different projections/roles.

The factor under test is the query-time StatePacket: can it reconstruct minimum
sufficient current state with provenance while opening fewer raw sources?

Do not migrate production memory stores for this experiment.

### Local SLM

Use z0intelligence #20 and Evolution Lab #23. The factor is a local learned
decision policy over an already-legal action set. Model variants are treatments
inside this factor, not extra top-level factors.

On the 12 GB target, do not co-reside multiple heavyweight SLMs. Measure one
candidate at a time and include deterministic/JEV/simple statistical controls.

### Bend

Use z0intelligence #48 and the downstream Bend semantic-attestation work as a
candidate structural verifier. Canonical AODL semantics remain authoritative.

Any unexplained Bend/canonical divergence is a hard failure, regardless of
latency.

## Phase 1 compositions

Only constituents that independently qualify may compose. Start with these
interaction questions:

- SoL-Pi × RLM: complementary compression/addressing or redundant?
- RLM × StatePacket: cheaper minimum-sufficient state?
- StatePacket × SLM: higher safe local coverage?
- SLM × Bend: does deterministic verification expand safe coverage?
- SoL-Pi × StatePacket: lower state-construction cost without evidence loss?

Every composite result must carry the constituent factor receipts. Never report
one blended "stack score" that destroys causal attribution.

## Metrics

Quality gates:
- independently verified success
- hard-policy violations
- provenance failures
- missed blocking unknowns
- stale-state use
- corrections/retries

Systems axes:
- input/context tokens
- context bytes
- raw source reads
- retrieval calls
- frontier calls
- p50/p95 latency
- local GPU-ms / peak VRAM
- dollar cost when actually measured

A token win cannot compensate for a verified-quality regression.

## Promotion

```
control
  -> singleton evidence
  -> qualified pairwise evidence
  -> higher-order composition
  -> shadow runtime
  -> assist
```

No upstream promotion is implied by any of these states.
