# FlyForge roadmap (maintainer-owned)

Revision binds to git SHA via `roadmap.yml`. Contributors **do not** rewrite this file. Issues labeled `needs-discussion` are not claims. Maintainers promote to `claimable`. Worker PRs follow `.verified-oss-loop/rollout.yml` (scheme `rolling`: day-pass → `preview`). P0 is locked on `main` ([#2](https://github.com/kvnloo/evolution-lab/issues/2) closed). TMNF-C is later ([#14](https://github.com/kvnloo/evolution-lab/issues/14)), not a claim.

Critical path: **lock a verified Hermes recovery contract, then run the control table under joules-per-verified-success.** Distill / grow / fly-Qwen search come after that table exists.

Registered product target (before more students): ≥95% of teacher verified success on the locked confirm split, ≤50% of teacher deployment cost, **zero** extra hard-policy violations. Hosted joules stay `unknown` until measured.

## P0 — Lock the contract (now)

GitHub [#2](https://github.com/kvnloo/evolution-lab/issues/2) · Linear [PER-1524](https://linear.app/0ism/issue/PER-1524)

Hermes recovery gym `hermes_recovery`. Actions `{retry, restart_sandbox, escalate, noop, page_human}`. Secrets fail closed. Direct-input is a mandatory skeptic. Tinker, FlyGym, and TMNF-C backends refuse.

Exit: failing unit test on secret-bearing fields; teacher closed-loop `gym-smoke`; locked train/val/confirm/ood splits on disk; a Verified OSS Loop receipt on the PR that asserts a Pareto point.

## P1 — Control table on Track A

MLP, GRU, direct-input, degree-preserving rewire, fixed reservoir. Kill criterion: if GRU and direct-input dominate a fly reservoir on val, do **not** spend a cycle on MaleCNS SGD. Pivot to motif students.

## P2 — Distill verified decisions

Teacher trajectories + outcomes, not plausible prose.

## P3 — Student states (DAgger) + bounded RL

Train on states the student visits. Fail-closed until Tinker is a real backend.

## P4 — Grow / reorganize

RigL / expansion only with function-preservation. After substantial rewrite, call it fly-inspired.

## P5 — Explain with interventions

Lesion delayed-cue; rewire refit; profile memory traffic.

## P6 — Freeze and confirm

Independent seeds, locked confirm, unseen failures. Then maybe Connectome-to-Function.

## Parallel (do not block P0)

- Motif distillation (EMD, loom, FlyHash)
- Native LM (`fly-hf`) only after `trust_remote_code` audit
- [kvnloo/fly-wirehead](https://github.com/kvnloo/fly-wirehead) pinout demo — not the student
- [kvnloo/TMNF-C](https://github.com/kvnloo/TMNF-C) (fork of adonis-singh/TMNF-C) — later visuo-motor gym; MaleCNS MB via DAN RPE; do not fork. [Issue #14](https://github.com/kvnloo/evolution-lab/issues/14)
- PER-944 Xenova WebGPU viewer — Linear Backlog, not this controller

## Sister repos

| Repo | Owns |
| --- | --- |
| [kvnloo/frontier-kb](https://github.com/kvnloo/frontier-kb) | Claims and protocol |
| [kvnloo/aodl](https://github.com/kvnloo/aodl) | HOTL 0.2 port, not a new kind |
| [kvnloo/verified-oss-loop](https://github.com/kvnloo/verified-oss-loop) | Claim lease / evidence receipt protocol |
| [kvnloo/fly-wirehead](https://github.com/kvnloo/fly-wirehead) | MaleCNS Shorts pinout demo (not the student) |
| [kvnloo/TMNF-C](https://github.com/kvnloo/TMNF-C) | Later TrackMania + MB gym (not this engine) |
