"""Bounded accounting projection of two retained studies; no inference or grading.

Usage: python stopped_audit.py GROQ_SOURCE CEREBRAS_SOURCE REPLAY_ROOT LEDGER
REPLAY_ROOT contains free-worker-ab/ and cerebras-ab/ from unchanged report.py.
Private inputs stay private. Standard Python only; do not use -O.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import uuid

FROZEN = {
    'free-worker-ab': {
        'protocol.json': 'bf7a8a7ee528ebb120c598f50eaa4fc008e0f872d1b355cdc18125c94dda5806',
        'canonical-receipts.jsonl': '81c32243e253205269c558563f47d07cbdcd702ad5fbda1bbf3b955d73346439',
        'summary.json': '54993dacbf80ce6355f974f0e3d34a54d4f893ead51f71b83c665ee4f5b1ed33',
    },
    'cerebras-ab': {
        'protocol.json': 'a7ede506c9cb2df01d84d041741a3dee81475a7ebff92a2cbad4eee725fbe348',
        'canonical-receipts.jsonl': '525d3a3792a973163dfd6c083fde3e9e3f8280efc068542626b00ff793db3dac',
        'summary.json': 'f0fe6c1ab37b7b01f851c8cfe5c3b3a5ab65ef5588b92ed4b30d48253809191a',
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(source, replay, ledger_lines, family):
    for name, expected in FROZEN[family].items():
        assert digest((source / name).read_bytes()) == expected, 'Wrong frozen source'
    protocol = read(source / 'protocol.json')
    pairs_files = sorted(source.glob('*/pair.json'))
    pairs = [read(p) for p in pairs_files]
    groups = {(p['replicate'], p['task']) for p in pairs}
    expected_count = 38 if family == 'free-worker-ab' else 4
    assert len(pairs) == len(groups) == expected_count
    assert groups == {(i, task) for i in range(expected_count // 2) for task in ['summary', 'gate']}
    lines = (source / 'canonical-receipts.jsonl').read_bytes().splitlines(keepends=True)
    assert len(lines) == len(set(lines)), 'Duplicate lifecycle row'
    assert set(lines) <= ledger_lines, 'Wrong authority ledger'
    assert (replay / 'canonical-receipts.jsonl').read_bytes() == b''.join(lines)
    rows = [json.loads(line) for line in lines]
    physical = {r['trace_id']: r for r in rows if r.get('execution') == 'live'}
    calls = [r for r in physical.values() if r['extra'].get('physical_call_attempted')]
    identities = {(r['provider'], r['model'], r['extra']['response_id']) for r in calls}
    assert len(identities) == len(calls) == expected_count
    for _, _, identity in identities:
        assert identity.startswith('chatcmpl-')
        uuid.UUID(identity.removeprefix('chatcmpl-'))
    latest = {r['trace_id']: r for r in rows}
    unresolved = [r for r in latest.values() if r['extra'].get('status') in ['started', 'acquired']]
    workers = [read(source / p.parent.name / 'worker.json')['result'] for p in pairs_files]
    report = read(replay / 'summary.json')
    original = read(source / 'summary.json')
    token_sums = {k: sum(r[k] for r in calls) for k in ['input_tokens', 'output_tokens']}
    assert sum(token_sums.values()) == report['worker_tokens_all_physical']
    assert sum(p['worker_tokens'] for p in pairs) == sum(token_sums.values())
    assert Counter(r['extra']['status'] for r in calls) == report['physical_statuses']
    assert report['completed_pairs'] == len(pairs)
    assert all(r['extra'].get('provider_reported_cost_usd') is None for r in calls)
    for p in pairs:
        for arm in ['A', 'B']:
            u = p[arm]['usage']
            assert p[arm]['uncached_parent_input_tokens'] == u['input_tokens'] - u['cached_input_tokens']
        assert p['parent_total_delta'] == (p['B']['uncached_parent_input_tokens'] + p['B']['usage']['output_tokens'] - p['A']['uncached_parent_input_tokens'] - p['A']['usage']['output_tokens'])
    parent_delta = sum(p['parent_total_delta'] for p in pairs)
    cache_delta = sum(p['B']['usage']['cached_input_tokens'] - p['A']['usage']['cached_input_tokens'] for p in pairs)
    grade_revision = None
    if family == 'cerebras-ab':
        assert not unresolved
        assert all(r['extra']['finish_reason'] == 'length' for r in calls)
        assert all(not w['output'] for w in workers)
        assert all(r['output_tokens'] == protocol['worker_max_tokens'] for r in calls)
        assert {k: v for k, v in original.items() if k != 'at'} == {k: v for k, v in report.items() if k != 'at'}
    else:
        error = read(source / '019-summary/worker-error.json')
        assert error['returncode'] == 1 and 'TimeoutError: timed out' in error['stderr']
        assert len(unresolved) == 2
        assert original['completed_pairs'] != len(pairs), 'Stale-report control no longer holds'
        revised = read(source / 'grading-v3/summary.json')
        assert revised['completed_pairs'] == len(pairs)
        for task, values in report['functions'].items():
            assert {k: v for k, v in values.items() if k not in ['parent_A_pass', 'parent_B_pass']} == {k: v for k, v in revised['functions'][task].items() if k not in ['parent_A_pass', 'parent_B_pass']}
        grade_revision = {
            'sha256': digest((source / 'grading-v3/summary.json').read_bytes()),
            'status': 'historically REPORTED, not regraded or independently accepted',
            'gate_parent_A_pass': revised['functions']['gate']['parent_A_pass'],
            'gate_parent_B_pass': revised['functions']['gate']['parent_B_pass'],
            'gate_worker_pass': revised['functions']['gate']['raw_worker_pass'],
        }
    return {
        'source_digests': FROZEN[family], 'producer_revision': None,
        'pairs_bytes_sha256': digest(b''.join(p.read_bytes() for p in pairs_files)),
        'worker_bytes_sha256': digest(b''.join((source / p.parent.name / 'worker.json').read_bytes() for p in pairs_files)),
        'provider': protocol['provider'], 'model': protocol['model'],
        'parent_model': protocol['parent_model'], 'parent_effort': protocol['parent_effort'],
        'frozen_file_hashes_match': sum(digest(Path(p).read_bytes()) == h for p, h in protocol['frozen_files'].items()),
        'frozen_file_count': len(protocol['frozen_files']),
        'completed_pairs': len(pairs), 'natural_task_population': False,
        'retained_summary_completed_pairs': original['completed_pairs'],
        'retained_summary_physical_calls': original['physical_calls'],
        'lifecycle_rows': len(rows), 'receipted_physical_calls': len(calls),
        'distinct_response_ids': len(identities), 'response_id_family': 'chatcmpl-UUID',
        'physical_statuses': dict(Counter(r['extra']['status'] for r in calls)),
        'finish_reasons': dict(Counter(r['extra']['finish_reason'] for r in calls)),
        'unresolved_orchestration_rows': len(unresolved),
        'all_attempt_coverage': 'UNKNOWN' if unresolved else 'retained cohort only',
        'worker_tokens': token_sums, 'empty_worker_outputs': sum(not w['output'] for w in workers),
        'provider_cost_known_rows': 0, 'full_cost': 'UNKNOWN',
        'parent_uncached_input_plus_output_delta': parent_delta,
        'parent_cached_input_delta': cache_delta,
        'parent_cache_neutral_sensitivity_delta': parent_delta + cache_delta,
        'parent_delta_plus_worker_input_output': parent_delta + sum(token_sums.values()),
        'revised_grade_report': grade_revision,
        'functions_original_grade_replay': report['functions'],
        'historical_free_only': protocol['free_only'], 'current_admission': 'NOT_ESTABLISHED',
        'independent_review': 'NOT_RUN', 'performance': 'NOT_COMPARABLE',
    }


if __name__ == '__main__':
    groq, cerebras, replay, ledger = map(Path, sys.argv[1:])
    ledger_lines = set(ledger.read_bytes().splitlines(keepends=True))
    result = {family: audit(source, replay / family, ledger_lines, family)
              for family, source in [('free-worker-ab', groq), ('cerebras-ab', cerebras)]}
    print(json.dumps(result, indent=2, sort_keys=True))
