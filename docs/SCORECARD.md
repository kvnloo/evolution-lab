# Results scorecard

Generated from committed result JSON. These are exploratory observations, not promotion approvals.
Baselines are taken from the same file and split. No across-encoding comparisons are inferred.
The summary uses the confirm split when present; other splits are shown in the details.
Verdicts use full-precision deltas: |delta| < 0.01 is TIES_FLOOR.

## Summary

| File | Arm | Split | Accuracy | Floor | Δ vs floor | Verdict | Caveats |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| recovery-compare.json | rule | confirm | 0.977 | 0.318 | +0.659 | BEATS_FLOOR | metric_caveat, not_measured (see section) |
| recovery-compare.json | mushroom | confirm | 0.839 | 0.318 | +0.521 | BEATS_FLOOR | metric_caveat, not_measured (see section) |
| shadow-real-traffic.json | ridge | confirm | 0.643 | 0.514 | +0.129 | BEATS_FLOOR | confidence_note (see section) |
| recovery-compare.json | majority | confirm | 0.318 | 0.318 | +0.000 | TIES_FLOOR | metric_caveat, not_measured (see section) |
| shadow-real-traffic.json | majority | confirm | 0.514 | 0.514 | +0.000 | TIES_FLOOR | confidence_note (see section) |
| shadow-real-traffic.json | mushroom_champion | confirm | 0.143 | 0.514 | -0.371 | LOSES_TO_FLOOR | confidence_note (see section) |
| recovery-llm-compare.json | hammer2.1_3b | confirm | 0.281 | — | — | NO_COMPARABLE_FLOOR | encoding_caveat (see section) |
| recovery-llm-compare.json | nanojev | confirm | 0.000 | — | — | NO_COMPARABLE_FLOOR | encoding_caveat (see section) |

## `results/recovery-compare.json`

- Task: hermes_recovery
- Metric: per-step action accuracy over all 8 steps
- Splits: confirm, ood
- Evidence class: exploratory_beta
- Promotion eligible: false
- splits: data/p0

| Split | Arm | Accuracy | Floor (majority) | Δ vs floor | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| confirm | majority | 0.318 | 0.318 | +0.000 | TIES_FLOOR |
| confirm | mushroom | 0.839 | 0.318 | +0.521 | BEATS_FLOOR |
| confirm | rule | 0.977 | 0.318 | +0.659 | BEATS_FLOOR |
| ood | majority | 1.000 | 1.000 | +0.000 | TIES_FLOOR |
| ood | mushroom | 1.000 | 1.000 | +0.000 | TIES_FLOOR |
| ood | rule | 1.000 | 1.000 | +0.000 | TIES_FLOOR |

### Caveats

- **metric_caveat:** Per-step accuracy is a proxy for the P0 closed-loop reward, not the same number. The P0 control table is quoted separately and must not be merged with these values.
- **not_measured:** NanoJev and Hammer3B are not in this table: they consume a typed state, not an 8x15 Hermes frame tensor, so including them needs a frame->state rendering that does not exist yet. Rendering it would be a new encoding, and comparing a fresh encoding against a policy fitted on the original features would not be a like-for-like comparison.

## `results/recovery-llm-compare.json`

- Task: hermes_recovery
- Metric: action accuracy (text-rendered inputs)
- Splits: confirm, ood
- Evidence class: exploratory_beta
- Promotion eligible: false
- Floor: not provided by this source; verdicts are NO_COMPARABLE_FLOOR. The different frame encoding prevents a controlled cross-file comparison.

| Split | Arm | Accuracy | Floor (majority) | Δ vs floor | Verdict | N | Unparsed | Latency p50 (ms) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| confirm | hammer2.1_3b | 0.281 | — | — | NO_COMPARABLE_FLOOR | 48 | 0 | 46.700 |
| confirm | nanojev | 0.000 | — | — | NO_COMPARABLE_FLOOR | 48 | 0 | 57.900 |
| ood | hammer2.1_3b | 0.000 | — | — | NO_COMPARABLE_FLOOR | 32 | 0 | 40.600 |
| ood | nanojev | 0.000 | — | — | NO_COMPARABLE_FLOOR | 32 | 0 | 58.400 |

### Caveats

- **encoding_caveat:** NanoJev and Hammer3B receive a text rendering of the frame tensor introduced by this script, not the 8x15 encoding the mushroom was fitted on. Values are quantised to 0-9 because NanoJev's encoder is capped at 512 tokens and refuses to truncate. Not a controlled bake-off; do not merge with recovery-compare.json.

## `results/shadow-real-traffic.json`

- Task: Hermes next-action shadow traffic
- Metric: held-out action-family accuracy
- Splits: confirm
- Evidence class: exploratory_beta
- Promotion eligible: false
- split: chronological
- dataset: next_action.jsonl (local path redacted)
- dataset_kind: live Hermes next-action log
- n_confirm: 1477
- n_train: 3444
- n_gold_rows: 4921

| Split | Arm | Accuracy | Floor (majority) | Δ vs floor | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| confirm | majority | 0.514 | 0.514 | +0.000 | TIES_FLOOR |
| confirm | mushroom_champion | 0.143 | 0.514 | -0.371 | LOSES_TO_FLOOR |
| confirm | ridge | 0.643 | 0.514 | +0.129 | BEATS_FLOOR |

### Caveats

- **confidence_note:** Mushroom confidence is derived as softmax over the MBON logits. The bundle has no calibrated head, so this is not a calibrated probability; it is monotone in the logit margin and is adequate to sweep coverage.

## Not rendered

- None.
