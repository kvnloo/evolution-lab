# Execution evidence

Base: `0212fee31a78cdba506f4d2684813962bfe0efb0`.
Branch: `factory/hermes/31-learning-integration-20261010`.
No publication, merge, GitHub comment, protected-evaluator change or model call.

Private evidence root (mode 0700):
`/workspace/zer0/data/operations/oss-factory/runs/20261010-hermes-z0-learning-integration/provider-worker/`

Each numbered gate has immutable `.json`, `.stdout`, `.stderr` files there. JSON
records the exact argv, cwd, exit, measured wall and user/system child CPU time.
The gates ran in `unshare --user --map-root-user --net` with no network route,
`PYTHONDONTWRITEBYTECODE=1`, model-hub offline variables and no visible CUDA device.
The CLI regression adds socket/import/process/function-call tripwires. These are
preparation-only synthetic receipts, not efficacy or independent verification.

## Vertical TDD

One behavior slice was tested RED, minimally implemented, then tested GREEN before
starting the next. Every row below used the **same cumulative command** on both
sides; the recorded failure is feature/guard absence, not a fabricated source run:

```sh
unshare --user --map-root-user --net python -B -m unittest discover \
  -s tests -p test_training_provider.py -v
```

| Slice / receipt prefix | Exit RED / GREEN | Wall seconds RED / GREEN | CPU seconds RED / GREEN |
|---|---|---|---|
| `01-real-interface` | 1 / 0 | 0.293232 / 0.170639 | 0.179898 / 0.161003 |
| `02-split-lineage` | 1 / 0 | 0.216944 / 0.184263 | 0.213933 / 0.180846 |
| `03-lineage-preflight` | 1 / 0 | 1.319125 / 1.179342 | 1.300037 / 1.162723 |
| `04-tool-semantics` | 1 / 0 | 1.547295 / 1.823213 | 1.501920 / 1.782510 |
| `05-manifest` | 1 / 0 | 2.131112 / 1.831217 | 2.111365 / 1.817408 |
| `06-text-shapes` | 1 / 0 | 2.731579 / 2.318805 | 2.635821 / 2.284467 |
| `07-cli` | 1 / 0 | 2.375807 / 3.073040 | 2.347964 / 3.038711 |
| `08-private-artifacts` | 1 / 0 | 3.202708 / 2.192682 | 3.162325 / 2.143770 |
| `09-source-identity` | 1 / 0 | 2.995356 / 2.980116 | 2.934537 / 2.944599 |
| `10-ambiguous-json` | 1 / 0 | 3.875783 / 4.248476 | 3.816538 / 4.189704 |
| `11-native-duplicates` | 1 / 0 | 3.793501 / 3.926158 | 3.751227 / 3.873224 |
| `12-native-alias` | 1 / 0 | 3.971697 / 3.936734 | 3.921697 / 3.891420 |
| `13-empty-native` | 1 / 0 | 3.590354 / 3.746290 | 3.494279 / 3.591400 |

The initial regression invoked the actual original extractor CLI on an owned
synthetic HOME before failing because the integration was absent. No production
adapter existed at that RED. Later REDs include partition leakage, missing
lineage, unsupported tools/text, mutable manifest, missing CLI output, permissive
file modes, hidden source symlinks, ambiguous JSON and duplicate/empty origins.
The additional equal-content/different-origin test is a post-GREEN control, not
a claimed new RED. Historical issue-31 negatives and the separate eight-cycle
report were not rerun, edited or relabeled as new findings.

## Bounded gates before commit

- `15-adjacent-GREEN` — Adjacent episode/grouped-split contracts: 19 passed. Gate exit 0; wall 0.067782s; CPU 0.065650s.
- `16-lint` — Ruff on the two authorized Python files. Gate exit 0; wall 0.108273s; CPU 0.013695s.
- `17-actual-cli-e2e` — Actual original extractor CLI -> preparation-only CLI; 2 train / 1 validation / 1 protected. Gate exit 0; wall 0.188151s; CPU 0.156907s.
- `18-type-check` — Initial mypy invocation: environment failure, not a code diagnostic. Gate exit 1; wall 0.038106s; CPU 0.031401s.
- `19-type-check-user-site` — Existing user-site mypy, isolated to adapter; no issues. Gate exit 0; wall 1.374693s; CPU 1.168385s.
- `20-focused-final-precommit` — Final focused suite: 14 passed, no skips. Gate exit 0; wall 4.103954s; CPU 3.880190s.
- `21-cli-refusal-precommit` — Actual tool-bearing extractor -> CLI refusal; child exit 2, no candidate. Gate exit 0; wall 0.186777s; CPU 0.176734s.

The first mypy launcher failed because `PYTHONNOUSERSITE=1` hid its existing
user-site package. The retry removed that variable **for mypy only** and passed;
nothing was installed or downloaded. Preserve both receipts.

## Exact-head handoff

After the local commit, `final-*` gates rerun the focused and adjacent tests,
Ruff, mypy and actual CLI acceptance/refusal against that committed head.
`final-head.json` outside Git records the full tested commit, changed-file hashes,
source cleanliness, gate results and artifact inventory. It is not self-embedded
in this commit. Read it before attributing any final-head result.

`synthetic-final/candidate/candidate.json`, `train.sft.jsonl` and `lineage.json`
are the retained, actual positive CLI output for the separate read-only consumer.
`synthetic-final/extractor-receipt.json` preserves the original extractor argv,
exit and timings; `cli-receipt.json` preserves the candidate CLI argv and result.
`tool-refusal-final/cli-receipt.json` retains the negative exit and rationale.
Earlier artifacts remain under `synthetic-e2e/` and `tool-refusal-precommit/`.

## Limits

- This is an untrained preparation candidate, not a model or a learning gain.
- Work grouping, native origin mapping and protected inventory remain producer
  assertions. Exact hashes/retained source files do not independently qualify
  provenance; real-history qualification and pinned UNIT verification are separate.
- Only plain Claude Code user/assistant text is supported. Tool-bearing input
  fails before candidate writes rather than silently losing tool semantics.
- Only train bodies are prepared. No original training API is called (it has no
  grouped validation argument); validation/protected remain withheld.
- Full-suite training/evaluation/runtime experiments were intentionally not run.
  No paid/model reviewer, no-mistakes publication pipeline or live evaluator ran.
- Mutation score n/a; energy unknown; no success/gain/cost-saving/promotion claims.
