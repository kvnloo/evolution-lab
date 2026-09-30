# T3b action-space v1 — pre-registration

Branch `exp/t3b-action-space-v1`. Written and committed **before** the v1 labels were
mined and before any model was scored. Anything changed after that commit is listed
under "Deviations" at the bottom with a reason.

## Why

T3b's first pass mapped real Hermes failure episodes onto the gym action space
`{retry, restart_sandbox, escalate, noop, page_human}`. Under the loose mapping the labels
collapsed to noop-vs-retry (1395 / 519 of 1914 labelled), with "moved on with another tool"
folded into noop. All learners landed on majority (0.69–0.71 test acc), so the comparison
said nothing about recovery control. v1 changes the action space to what agents actually do.

## Action space

`ACTIONS_V1 = (noop, retry, edit_retry, switch_tool, ask, abort)`

| action | meaning |
|---|---|
| noop | carry on with the plan; the failure is not addressed |
| retry | re-issue the same call unchanged |
| edit_retry | change something, then re-attempt the same thing (edited args, same target, or repair the environment) |
| switch_tool | go after the same target with a different tool or another agent |
| ask | hand control to a human (clarify tool, or stop and talk to a present human) |
| abort | end a headless run reporting that it could not be done |

## Episode and privacy

Same episode definition and ids as `scripts/t3b_mine_hermes.py`: one failed tool
result + the agent's next assistant turn + what happened after. The miner reads
`~/.hermes/state.db` read-only. Raw text is reduced in-process to:

* per call in the next 3 turns: tool family, same-tool flag, identical-args flag,
  token Jaccard with the failed call's args, containment of the failed call's salient
  argument **value** tokens (len>=4, non-numeric, stop-listed) in this call's tokens (`ovl`),
  count of shared path-like tokens (`path_shared`), restart / clarify / delegate flags,
  success flag;
* per turn with no calls: one bit — the text matches `FAIL_REPORT`
  (`unable to|couldn't|can't|cannot|failed|not possible|blocked|denied|did not succeed|wasn't able|not able to|unsuccessful|aborting|giving up|gave up`).

Private store: `~/.z0int/research/t3b/episodes_v1as.jsonl` (0600). Git gets code, hashed
ids never, and aggregate counts / metrics only.

## Labelling rule (code: `evolution_lab/action_space_v1.py::taken_action`)

At the first assistant turn N after the failure; first match wins.

| id | condition | action |
|---|---|---|
| R0 | no turn N, or a human message precedes N | excluded (not an agent decision) |
| R1 | N has no tool calls | interactive source: **ask**; headless: **abort** if `FAIL_REPORT` matches N's text, else **noop** |
| R2 | any `clarify` call in N | ask |
| R3 | any `delegate_task` / `a2a_call` in N | switch_tool |
| R4 | any restart-type call in N (process kill/restart, `systemctl restart`, `docker … up`, …) | edit_retry |
| R5 | same tool, identical args | retry |
| R6 | same tool, token Jaccard >= 0.5 | edit_retry |
| R7 | same tool, related target (`ovl >= 0.25` or `path_shared >= 1`) | edit_retry |
| R8 | different tool, related target | switch_tool |
| R9 | otherwise (nothing in N touches the failed target) | noop |

### Outcome verifier ("did it recover", `recovered`)

Any action with `session_complete is False` → not recovered. Otherwise:

* retry / edit_retry: a same-tool, non-restart call on the same target (identical, Jaccard>=0.5,
  or related) succeeds within turns N..N+2;
* switch_tool: at least one of N's matching calls (delegate or other-tool related) succeeded;
* ask: clarify call succeeded, or (interactive stop) the human replied at N+1; else unverifiable;
* abort: the run ended cleanly (`session_complete`);
* noop: session completed and no failed same-tool related re-attempt in N..N+2.

**Primary label** = taken action iff verified recovered, else unknown (excluded).
**Sensitivity label** = taken action regardless of outcome (behavioural cloning).

### Audit (pre-registered)

Stratified sample, seed 20260930: up to 12 episodes per behavioural action (all if fewer).
The auditor sees only the structural skeleton (no text), without the rule output, and
labels each episode by the action definitions above; we report raw agreement and Cohen's
kappa vs the rule. Known limitation, stated up front: a structural audit checks that the
rule implements the definitions and catches ambiguous skeletons; it cannot check the
semantics that only the text shows (e.g. whether an unrelated next call really means
"moved on"). A content audit needs the owner to read raw text locally:
`scripts/t3b_audit_v1as.py content` writes a private worksheet under `~/.z0int`, and
`score` reports agreement as aggregates only.

## Comparison (code: `scripts/t3b_eval_v1as.py`)

Data: v1 episodes whose lineage is in the frozen `split_v1` (train/val/test by salted
lineage hash; unchanged). Episodes from lineages created after the split are dropped.

Features (failure-time only; nothing from turn N or later): `v1_frame` = 10 gym base fields +
previous v1 action in the session (strictly earlier turn) + source / tool family / exit bucket
one-hots + counts (42 dims); `gym` subset = first 16 dims. Both go through the incumbent PN
encoder (`models._pn_features`, 8-frame lineage history).

Controls and candidates:

1. `majority` — train mode.
2. `best_constant` — the constant action with the lowest mean train cost.
3. `rule_v1` — hand-written, failure-time fields only, never aborts (`action_space_v1.rule_v1`).
4. `ridge_{gym,rich}` — alpha in {0.1,1,10,100} chosen by val mean cost.
5. `mlp_{gym,rich}` — 64 hidden, alpha in {1e-4..1e-1} by val mean cost; seeds 0..4 (seed 0 reported, spread shown).
6. `mb_{gym,rich}` — incumbent local-plasticity mushroom body (frozen sparse PN→KC, k-winner KCs,
   only KC→MBON plastic, `plasticity_train_k` = reference update rule with 6 outputs), incumbent
   gene (128 KC, k=13, lr 0.35, 20 epochs), seeds 0..4.
7. `mbtuned_{gym,rich}` — same, gene chosen on val cost from {128,512} KC × lr {0.1,0.35} × epochs {10,20}.

Decoding: every model with scores is reported twice — `argmax`, and `cost` = Bayes decision
under the cost matrix using softmax(scores/T), T chosen on val NLL from {0.25,0.5,1,2,4}.

Cost matrix `C[true, pred]` (`action_space_v1.COST`): 0 on the diagonal, 1 otherwise, except
predicting **abort** when it is wrong = 5; **missing an ask** (true ask, predicted anything else
but abort) = 4; a needless ask = 1.5; a **redundant retry** (true edit_retry or switch_tool,
predicted retry) = 0.5, and edit_retry when a plain retry was right = 0.5.

Metrics on test: mean cost (primary), accuracy, macro-F1, balanced accuracy, per-class recall
(reported as n/a when a class has <5 test examples, but always inside the cost).
CIs: lineage-grouped bootstrap, 2000 resamples, seed 0; paired differences on the same resample.

**Primary endpoint.** `mb_rich` seed 0, cost decoding, vs `best_constant`:
95% CI of (cost_MB − cost_const). MB "beats the trivial controller" iff the upper bound < 0.
**Secondary.** Same diff for every model vs `best_constant` and vs `rule_v1`; `mb_rich` vs
`mlp_rich` (both cost-decoded): MB is "competitive" iff the upper bound of (cost_MB − cost_MLP) < 0.05.
**Sensitivity.** (a) behavioural labels; (b) plain 0/1 cost (= error rate).

A null (CI straddling 0) is reported as a null. The test split is read once, by the final run.

## Deviations

(none at commit time)
