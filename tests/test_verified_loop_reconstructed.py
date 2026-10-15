"""verified_loop_reconstructed: synthetic tables only (no real data)."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from evolution_lab import verified_loop_reconstructed as vr
from test_verified_loop import synthetic_graded


def table(n, groups, seed, *, signal=True, reconstructed=True, prefix='g'):
    """v0's graded generator: risk lives in a feature the gate ignores; the gate ASKs on a random 20%."""
    rows = synthetic_graded(n, groups, seed, signal=signal)
    rng = random.Random(seed + 100)
    for r in rows:
        r['group'] = prefix + r['group']
        if rng.random() < 0.2:
            r['gate'] = 'ASK'
            r['features']['gate=ACT'], r['features']['gate=ASK'] = 0, 1
        if reconstructed:
            r['provenance'] = 'reconstructed'
    return rows


def write(d, name, rows):
    p = Path(d) / name
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return p


class ReconstructedLoop(unittest.TestCase):
    def run_tables(self, train, held, lives=()):
        with tempfile.TemporaryDirectory() as d:
            lp = [write(d, f'live{i}.jsonl', rows) for i, rows in enumerate(lives)]
            return vr.run(write(d, 'train.jsonl', train), write(d, 'eval.jsonl', held), lp, steps=200, boot_b=200)

    def test_pooling_observed_rows_into_a_reconstructed_table_is_refused(self):
        train = table(50, 10, 1) + table(5, 2, 2, reconstructed=False)
        with self.assertRaises(vr.Refused):
            self.run_tables(train, table(10, 5, 3))
        with self.assertRaises(vr.Refused):  # a live table may hold observed rows only
            self.run_tables(table(50, 10, 1), table(10, 5, 3), [table(10, 5, 4)])

    def test_small_tables_are_insufficient(self):
        res = self.run_tables(table(60, 5, 1), table(20, 3, 2))
        self.assertEqual(res['verdict'], 'INSUFFICIENT_DATA')

    def test_a_reconstructed_win_without_sufficient_live_rows_is_only_pending(self):
        res = self.run_tables(table(900, 40, 1), table(600, 30, 2), [table(40, 4, 3, reconstructed=False)])
        self.assertEqual(res['stage_a_train_cv']['decision'], 'SHADOW_CANDIDATE')
        self.assertTrue(res['stage_b_eval_window']['passed'])
        self.assertEqual(res['reconstructed_verdict'], 'RECONSTRUCTED_PROPOSAL')
        self.assertEqual(res['verdict'], 'PENDING_LIVE_VALIDATION')

    def test_shadow_candidate_needs_a_sufficient_passing_live_cohort(self):
        res = self.run_tables(table(900, 40, 1), table(600, 30, 2),
                              [table(600, 30, 5, reconstructed=False, prefix='L')])
        self.assertEqual(res['verdict'], 'SHADOW_CANDIDATE')
        res = self.run_tables(table(900, 40, 1), table(600, 30, 2),
                              [table(600, 30, 5, reconstructed=False, signal=False, prefix='L')])
        self.assertEqual(res['verdict'], 'NO_IMPROVEMENT_ON_LIVE')


if __name__ == '__main__':
    unittest.main()
