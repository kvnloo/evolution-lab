# External provider: preparation only

Issue #31. This is an offline data-preparation seam, **not a trainer**, model,
evaluator, scheduler, provider SDK, live route or promotion mechanism.
Source attribution: **sybil-solutions / SouthpawIN**. The original extractor and
formatting algorithms are reused unchanged, not reimplemented or vendored.

## Pinned interfaces

| Source | Revision | Interface |
|---|---|---|
| evolutionary-training | `68ecf2057586a232bea45ad0fafcdfdfae56089f` | `scripts/mega_training_data.py:format_to_sharegpt(row, dataset_key)` |
| same | same | `scripts/agentic_training_loop.py:format_for_sft(data)` |
| ai-data-extraction | `9176c2dd81c8a101b52debd4cec88573284e2b30` | actual `extract_claude_code.py` CLI |

Local mirrors used for these gates are `kvnloo/evolutionary-training` and
`kvnloo/ai-data-extraction`. Full script SHA-256s are embedded in the adapter and
emitted in each candidate. HEAD, tracked/nonignored worktree cleanliness, exact
Git blob bytes and script SHA-256 are checked **before import**. Symlink source
substitution is rejected, including changes hidden by skip-worktree. Verified
source bytes are compiled into isolated module namespaces, never loaded from
provider `sys.path` entries or cached bytecode. Ignored extractor caches are not
executed or removed. Source checkouts must not be mutated concurrently.

Only the two formatting functions are called. Do **not** invoke the global
preparation CLI (its output option and output name do not match the consumer),
`training_loop.main`, health checks (including their POST fallback), training,
evaluation, downloads or installation. No models or provider clients are loaded.
`provider.json` is an exact declarative allowlist, **not executable commands**.
Unknown keys, alternative phases/commands/revisions, judges, gates and every
verifier configuration (including an unpinned verifier) are refused. Independent
pinned UNIT verification belongs to a separate read-only consumer, not this
manifest. The protected evaluator is not invoked or modified.

## Public API and CLI

```python
from pathlib import Path
import json
from evolution_lab.training_provider import PreparationError, prepare_candidate

result = prepare_candidate(
    provider_root=Path(provider_checkout),
    extractor_root=Path(extractor_checkout),
    corpus_path=Path(actual_extractor_jsonl),
    lineage_path=Path(explicit_lineage_json),
    output_dir=Path(new_private_directory),
    manifest=json.loads(Path(provider_manifest).read_bytes()),
)
assert result["status"] == "untrained"
```

`prepare_candidate(...) -> dict` writes the files described below. Domain refusals
raise `PreparationError(ValueError)`; filesystem/JSON/Git errors also abort the
candidate. `main(argv: list[str] | None = None) -> int` is the executable seam.
Importing the adapter has no provider execution or artifact-writing side effects.

```sh
# Checkouts and corpus/lineage must already exist. No automatic install/download.
# OUTPUT must not exist, its parent must exist, and it must be outside sources/Git.
umask 077
unshare --user --map-root-user --net \
  python -B -m evolution_lab.training_provider --dry-run \
  --provider-root "$PROVIDER" --extractor-root "$EXTRACTOR" \
  --corpus "$CORPUS" --lineage "$LINEAGE" --output "$OUTPUT" \
  --manifest docs/experiments/training-provider-integration-20261010/provider.json
```

The CLI requires `--dry-run` and returns JSON on stdout; provider formatting logs
go to stderr. Domain refusals return 2 and explain the reason on stderr.
There are no train/eval/download/install switches. The network namespace above
is an external verification control; the API itself is not an OS sandbox.

## Source lineage contract

Input JSON object, schema `evolution_lab.source_lineage.v1`, with exactly:

- `qualification`: literal `producer_asserted`.
- `corpus_sha256`: SHA-256 of the exact extractor JSONL file bytes.
- `protected_sessions`: explicit list of protected native session identifiers,
  including an explicit empty list when there are none. Unknown inventory entries
  or a mismatch with the supplied protected partition are refused.
- `records`: one ordered binding per JSONL record, with these exact fields:
  `record_index` (zero-based integer), `record_sha256` (exact JSONL line bytes,
  **including its line ending**), `source_file` (original absolute native path),
  `source_sha256` (exact native file bytes), `session_id`, `work_item_id`, `split`.
  `split` is exactly `train`, `validation` or `protected`.

`schema` is also required. No implicit grouping, default bucket, missing binding,
unknown split or duplicate JSON object key is accepted. Native source files must
exist and have matching bytes; the session must match both extractor output and
native filename. Duplicate native file origins (including path aliases) are
refused. Session IDs, producer-asserted work-item IDs and canonical native file
identities must not overlap partitions. Identical source bytes across partitions
are also refused as a leakage control; **a hash is never an origin identifier**.
Equal bytes from distinct native origins in one partition retain both mappings.

Whole groups are assigned **before preparation**, using the existing `SessionSplit`
(`train`, `dev` = validation, `sealed` = protected; no future bucket here). The
adapter does not invent chronology or reassign groups. Only train rows reach
original preparation functions. Withheld bodies are never put in candidate data.

The supported slice is plain Claude Code user/assistant text with both roles
present. Unknown extracted row/message fields, alternate converter formats,
unsupported roles, structured message content, non-text native content blocks and
unknown native event types are refused. Tool-bearing native or extracted records
are refused across **all** partitions before any artifact writes. This includes
separate tool results, orphan results otherwise dropped by the extractor,
`tool_use(s)`, tool roles, function calls and multiple `tool_calls`. The error
explains that original preparation loses tool-event semantics. This adapter does
not flatten or repair them. Ordinary metadata such as model/timestamp is not
training content; its authoritative bytes remain in the original hashed sources.

**These checks validate bytes and internal consistency, not historical truth.**
The work-item mapping, corpus-to-source binding and protected inventory are
producer assertions. Renaming an origin or omitting a genuinely protected source
cannot be disproved from this sidecar alone. Retaining source files/hashes is not
independent evidence that those identifiers came from real history. Independent
source qualification must happen outside this seam before any future training
or evaluation authorization. The adapter never marks lineage independently
qualified and never claims real-history evaluation or learned improvement.

## Outputs and consumer rules

A fresh mode-0700 directory contains mode-0600 files:

1. `train.sft.jsonl`: original preparation/formatter output, train partition only.
2. `lineage.json`: original bindings and protected inventory retained, plus
   `independently_qualified: false`; each record has `prepared_index` and
   `prepared_sha256` for its exact output line, or nulls when withheld.
3. `candidate.json`: schema `evolution_lab.training_candidate.v1`, written last;
   status `untrained`, `model_artifact: null`, eligibility
   `preparation_only_not_training_authorization`, partition counts, `SessionSplit`,
   input digests, pinned source digests and output file hashes/byte lengths.
   `verified_task_success`, `verified_model_gain`, `verified_cost_saving` and
   `promotion` are all explicitly false under `claims`.

Existing output paths are never overwritten; output inside any of the three
source repositories is refused. Use a private root outside **all** Git checkouts.
On preflight refusal no candidate directory is created. On an I/O failure an
incomplete directory may remain; never treat its existence as success. A consumer
must parse the final manifest and verify artifact hashes and per-row linkage.
Manifest/source identity checks here do not authenticate the adapter itself: an
independent consumer must pin the tested Evolution Lab commit and verifier bytes.

## Reproduce bounded verification

The new test module invokes the actual extractor CLI with a synthetic-only HOME,
then exercises real functions and the real candidate CLI. Set
`EL_TRAINING_PROVIDER_SOURCE` and `EL_EXTRACTOR_SOURCE` to the pinned checkouts if
the documented local defaults are absent. Missing checkouts skip the integration
class (not proof); dirty or wrong pins fail. The recorded gate has no such skips.

```sh
unshare --user --map-root-user --net python -B -m unittest discover \
  -s tests -p test_training_provider.py -v
unshare --user --map-root-user --net python -B -m unittest \
  tests.test_episode tests.test_capability_miner.SealedSplit \
  tests.test_capability_miner.Schema -v
ruff check --no-cache evolution_lab/training_provider.py tests/test_training_provider.py
```

The CLI test also installs process-local socket/import/process/function-call
tripwires. No full model/training/evaluation suite is authorized. These are
synthetic preparation and boundary tests, not model quality, task-success,
latency-saving or cost-saving evidence. Mutation score: n/a; energy: unknown.
See `EVIDENCE.md` for the recorded RED/GREEN and actual CLI artifact locations.
