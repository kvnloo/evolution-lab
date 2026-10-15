# Pre-registration addendum: verified loop v0 on RECONSTRUCTED strata

Status: **registered before any reconstructed result is scored.** This file is the first commit on branch
`exp/verified-loop-reconstructed-v0` (off `exp/verified-loop-v0`). The runner and any result come in later commits.
Changing a rule below after the first result needs a new pre-registration (v1) in its own commit. The v0 result
stays reported against v0.

This is an **addendum** to `docs/prereg/verified-loop-v0.md`. Everything v0 fixes is unchanged and applies here:
the question, the features, the label, the analysis set, the model and its constants, the policies `G`, `L_cov`
and `L_crc`, grouped K = 5 cross-validation, the cluster bootstrap, the sufficiency gate and the decision rule. The
addendum adds four things only: where reconstructed data may be used, a temporal/lineage split, a held-out check,
and a rule that a reconstructed-trained policy is only a proposal until it is validated on LIVE observed rows.

Context: z0intelligence#143 lane D, #54, #56. Data producer: z0intelligence branch `feat/sft-replay-reconstruct`,
module `z0int.replay_reconstruct` 0.1.0.

## What reconstructed data is

Live rows (`provenance` absent; called "observed") carry a DecisionOpportunity built at prompt time. Reconstructed
rows are rebuilt later from AgentsView history and transcripts. Their opportunity is rebuilt from the request text
against a State Packet with **no claims** and **git state unknown**: git facts at that time were not recorded and
are never guessed. Their labels come from the same outcome verifier (GitHub disabled). Each reconstructed row has
`provenance: "reconstructed"` and a label class under `reconstructed/`.

Known consequence (accepted, reported): in a reconstructed row every git fact family the request needs is a
blocking `source_unavailable` unknown. So `legal.ACT` and `gate=*` are often more conservative than in a live row
for the same request. This is a feature-distribution shift between reconstructed and live rows. It is the main
reason why stage C below (live validation) is mandatory.

## Strata: a separate learner run each, never pooled

- One stratum per harness and cohort: `<harness>/reconstructed` (user turns of root sessions) and
  `<harness>/reconstructed-agent` (turns of subagent sessions). Each stratum gets its own learner run.
- A reconstructed stratum is never merged with an observed table, with another reconstructed stratum or with
  another harness. The runner refuses (exit with an error, no result) when:
  - a train or eval table has a row whose `provenance` is not `reconstructed`;
  - a live table has a row whose `provenance` is `reconstructed`;
  - the train, eval and live tables have different `feature_schema_sha` / `table_version`.
- Features: the `features` object only, which is the live v0 vocabulary. A policy fitted on reconstructed rows can
  therefore be applied to live rows without translation. The `context`, `lineage` and `observed` objects of a
  reconstructed row are never model inputs.
- Analysis set, label, groups: exactly as v0. `group` is the hashed root session, so subagent and continuation
  sessions share the lineage of their root.

## Split (temporal + lineage, as `z0int.episode_split`)

- Eval window **locked to [2026-10-07, 2026-10-15)**, which is 2026-10-07..2026-10-14 with both days included.
  This is the window the owner asked for. I found no reason to change it. The end is exclusive so that the partial
  current UTC day (2026-10-15, the run day) falls in `recent` and not in eval.
- Dev = the 7 days before eval ([2026-09-30, 2026-10-07)). Train = days before 2026-09-30. Recent = from 2026-10-15.
- Rules, in order (from `episode_split`, applied to reconstructed episodes only):
  1. Undated rows are quarantined.
  2. A lineage group lives in one split only. Its earlier parts are quarantined.
  3. A content identity (normalised request hash, only for requests of 32 normalised characters or more) is owned
     by the highest-priority split (eval > dev > recent > train).
  4. Within a split, only the first row per semantic hash is kept.
- **Live-lineage quarantine:** a group that has live opportunity records on the host is removed from train and
  dev. It shares lineage with the live validation pool and must never be fitted on. It may stay in eval and recent,
  because those windows never feed fitting.
- The lock file is written once by the first split. A different window needs a new lock and a new
  pre-registration. The window is never moved after a result is seen.

## Stages and decision rule

**Stage A, train (v0 unchanged).** The v0 analysis, v0 sufficiency gate and v0 decision rule run on the
stratum's **train** table only. Result: `A ∈ {INSUFFICIENT_DATA, NO_IMPROVEMENT, SHADOW_CANDIDATE}`. The v0
secondary sensitivity analysis is not run here; it is not decisional.

**Stage B, held-out reconstructed eval window.** One frozen policy is fitted on the full train analysis set: the
v0 model, and `tau_cov` set on train so that the train ACT coverage of `L_cov` matches `G` (v0 rule). It is applied
unchanged to the analysis set of the stratum's **eval** table.
- Sufficiency (the same counts as v0): at least 300 rows, at least 30 not-success rows and at least 10 groups,
  with both classes present. Held-out data has no folds, so the K = 5 fold criterion does not apply.
- Criteria: `delta_eval = risk(L_cov) - risk(G) < 0`; the 95% cluster-bootstrap CI upper bound is `< 0` (B = 2000,
  seed 0, over eval groups); and zero rows where `L_cov` ACTs while `legal.ACT == 0`. ACT coverage on eval is
  reported but is not a criterion, because a threshold fixed on train can drift.

**Verdict from the reconstructed data:**
- `INSUFFICIENT_DATA` if stage A or stage B fails its sufficiency gate;
- `NO_IMPROVEMENT` if A is `NO_IMPROVEMENT` or B fails a criterion;
- otherwise `RECONSTRUCTED_PROPOSAL`. This is **not** a SHADOW_CANDIDATE. On reconstructed evidence alone a policy
  can only ever be proposed for shadow, never shadowed or promoted.

**Stage C, LIVE validation (required before any promotion request).** The same frozen policy (model plus
`tau_cov`) is applied, unchanged, to each LIVE observed table of the same harness: `<harness>/<cohort>` from the
live loop export. Each cohort is evaluated separately and never pooled. Nothing is refitted and no threshold moves.
Live rows are used only here, never for fitting.
- Sufficiency and criteria: the same as stage B.
- `SHADOW_CANDIDATE` (live-validated) only if the verdict is `RECONSTRUCTED_PROPOSAL` **and** at least one live
  cohort passes the stage C sufficiency gate and all stage C criteria. Even then it earns **shadow only**: the
  existing `promotion_request.v0` path with `requested_stage: shadow`, the #56 contract fields and explicit owner
  approval. Going past shadow needs online shadow evidence under a separate registration, exactly as in v0.
- `NO_IMPROVEMENT_ON_LIVE` if a live cohort is sufficient and the policy fails a stage C criterion there.
- `PENDING_LIVE_VALIDATION` if no live cohort is sufficient. The proposal waits and changes nothing.

Final outcomes are exactly one of `INSUFFICIENT_DATA`, `NO_IMPROVEMENT`, `PENDING_LIVE_VALIDATION`,
`NO_IMPROVEMENT_ON_LIVE`, `SHADOW_CANDIDATE`.

## Invariants

- A reconstructed-trained policy never sets or changes the live learner's thresholds, the deterministic gate or
  any harness configuration.
- Reconstructed results are written beside the live results and never replace them. The live loop tick never
  reads reconstructed tables.
- Every result records the train, eval and live table sha256, the split id, the build id and the learner commit.
- Report counts and metrics only. No rows, ids or text.

## Known threats (accepted for this addendum)

- Feature shift between reconstructed and live rows (git unknown, no packet claims). Stage C is the mitigation.
- Verification is missing not at random, as in v0. This is stronger for harnesses whose shell results carry
  exit codes only on failure (`failure_only`) or never (`none`). The table manifest records this.
- Old sessions come from a different tool mix, prompt style and model versions. The temporal split is the
  mitigation, and it is also why a reconstructed policy is never validated on reconstructed data alone.
- The revert/fix window is 7 days. For eval-window rows from 2026-10-08 on it has not fully elapsed on the run day,
  so a later revert or fix of those commits is not yet visible. Commit survival is a low-confidence soft signal and
  never decides a label on its own. Reverts and fixes already seen are kept. Rows from the run day itself fall in
  `recent`, outside every decisional split.
