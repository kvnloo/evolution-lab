# External training provider: two bounded preflight falsifications

Research receipt for [Evolution Lab #31](https://github.com/kvnloo/evolution-lab/issues/31), in the Hermes backlog lane. This is not a production provider adapter, trainer, dataset converter, capability judge, or promotion verdict. Both hypotheses remain **RED**. The full issue is not completed.

## Sources and ownership

- Evolution Lab reference: `0212fee31a78cdba506f4d2684813962bfe0efb0`, the current `nightly` worker base when this study started.
- External extraction/skill library: `kvnloo/ai-data-extraction` at `9176c2dd81c8a101b52debd4cec88573284e2b30`; original work by sybil-solutions.
- External trainer: `kvnloo/evolutionary-training` at `68ecf2057586a232bea45ad0fafcdfdfae56089f`; Gym Trainer/training machinery is SouthpawIN's work.
- Exact original file digests are in each cycle receipt. The libraries stay in separate checkouts, are imported/invoked directly, and were not edited, copied, or replaced.
- Scope is this experiment directory only. Claude's extractor files and Codex's independent z0evals/runtime work are excluded. Prior EL-23-S skill generation remains DISCARD and was not repeated.

## Cycle 001: direct extraction-to-training interoperability

Commit: `8cb2d35e21df74ed6549d5ad721b234ffb3cd43a`.

The unchanged `extract_codex.py` CLI ran in an isolated synthetic HOME. Its five explicitly constructed events produced one extracted conversation, two messages, two tool events, and retained session identity. The original `corpus_to_skills.sample_corpus` selected 1/1 records. No skill-generation inference was requested.

The original `agentic_training_loop.format_for_sft` produced **zero** training conversations from that extracted record. It expects `conversations` with `from`/`value`, whereas this extractor produces `messages` with `role`/`content`. This falsifies direct interchangeability, not the usefulness of either library.

Positive native-format control: one ShareGPT conversation produced one training conversation with user/assistant/tool roles. No converter was implemented to manufacture a GREEN result. Any future thin representation adapter must preserve source/session/work-item lineage and tool semantics while excluding protected evaluation material; that acceptance remains open.

Receipt: `cycle-001-data.json`. Expected command exit: 1; two tests, one failed hypothesis, one passing control, zero errors.

## Cycle 002: advertised dry-run versus no-inference preflight

Commit: `f2eb54b385fc435ca8971e5798977068bd1701fd`.

The original `training_loop.main()` ran with `--dry-run`. Only its endpoint configuration was redirected to an owned loopback HTTP fixture. After `GET /health` returned 503, the real call path attempted **one POST to `/v1/chat/completions`**. The fixture rejected it with HTTP 405 before inference. The provider describes this run as “checking model health only,” but its readiness fallback can request model inference.

Positive control: an owned endpoint returning health HTTP 200 produced exactly one GET and zero POSTs. Source-pin negative: a real different repository supplied as the trainer root was rejected with exit 2 before module import, and no receipt was created.

Receipt: `cycle-002-preflight.json`. Expected command exit: 1; two tests, one failed hypothesis, one passing control, zero errors.

Do not select this advertised dry-run as an unattended **zero-inference/read-only** preflight. Existing original `--help` commands were executed successfully for command resolution without training or inference. Hardware readiness, safe data eligibility, provider phase permissions, and actual training are separate, unqualified gates. No source-library fix was attempted.

## Independent reproduction and evaluation boundary

A separate read-only reviewer reproduced commit `f2eb54b385fc435ca8971e5798977068bd1701fd`: exit 1, four tests, two failed hypotheses, two positive controls, zero errors/skips. Its observations and original file hashes exactly matched both committed cycle receipts. No additional high/medium scoped review defect was found.

The sanitized replication is `independent-reproduction.json`; original private receipt digest is `8964310f289929ed7df9aea2b9eab2399b3a1920a6ef8f0b784605193887c033`. This is independent replication, **not** the canonical sealed z0evals #92/#93 verdict. Synthetic observations are not eligible observed task outcomes or proof of model-quality improvement. Safety booleans describe this inspected fixture run; they are not a general dispatch monitor.

## Reproduce safely

Set `EXTRACTOR_ROOT` and `TRAINER_ROOT` to separate original checkouts at the exact revisions above. Set `PRIVATE_RECEIPT` to a private path outside Git, and `TMPDIR` to an owned scratch directory. No private conversation history is needed.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 docs/experiments/training-provider-20261010/probe.py \
  --extractor-root "$EXTRACTOR_ROOT" --trainer-root "$TRAINER_ROOT" \
  --cycle all --receipt "$PRIVATE_RECEIPT"
```

Expected: exit 1, four tests, exactly two intentional hypothesis failures. `--cycle data` and `--cycle preflight` run the two bounded cycles separately. The source-specific extractor child has a ten-second timeout; execute the entire probe with a 45-second outer timeout. An unexpected error, extra failure, different source revision, or unexpected pass requires investigation, not relabeling.

## Coverage, cost and remaining blockers

- Initial two probe windows: 1.0446463389671408 s wall, 0.040969 s process-plus-child CPU. Independent replication: 1.0405897419841494 s wall, 0.032837 s process-plus-child CPU. These exclude startup/import/pin checks, coordinator/reviewer/tool-server cost. RSS is a process-lifetime peak, not aggregate memory.
- Probe-local inference calls and training runs: zero. Coordinator and reviewer used hosted subscription-authenticated models; these are **additional** costs, not zero-compute research. `cost-coverage.json` retains a partial coordinator counter interval and seven reviewer API calls. Whole-task model usage, invoice cost, hosted/local joules and GPU-ms are not fully measured. No compute-efficiency improvement or paired savings is claimed.
- Owning unit suite: reference and docs-only candidate each attempted 256 tests and returned exit 1, eight identical missing-JAX errors, two skips. The full suite is **not green**. No dependency install hid this limitation.
- Existing recovery gym `--n 8` returned exit 0. This is a toy/simulation control, not original-provider training, real agent task quality, or promotion evidence. Details: `gates.json`.
- No CI success is claimed. No PR was opened because the configured preview/nightly path may automerge; only a dedicated research ref is intended for publication.
- Next eligible EL #31 work: versioned thin provider manifest and fail-closed command resolution; explicit data representation/lineage boundary; protected split and judge-mutation negatives. Then independently qualify a genuinely eligible low-cost training arm. Existing library and Gym Trainer machinery remain the implementation substrate.
- No upstream post, default-branch merge, production routing/model changes, model download, raw-history publication, candidate skill installation, replacement scheduler, corpus store, trainer, or mutable evaluation framework occurred.
