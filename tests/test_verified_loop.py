"""verified_loop v0: synthetic tables only (no real data)."""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from evolution_lab import verified_loop as vl

FEATS = ['scope_mode=question', 'n_unknowns', 'risky', 'legal.ACT', 'legal.ASK', 'legal.OBSERVE',
         'legal.ABSTAIN', 'gate=ACT', 'gate=ASK']


def row(i, group, y, *, risky=0, legal_act=1, gate='ACT', observed='ACT', state=None):
    f = dict.fromkeys(FEATS, 0)
    f.update({'risky': risky, 'legal.ACT': legal_act, 'legal.ASK': 1 - legal_act, 'legal.ABSTAIN': 1,
              f'gate={gate}': 1, 'n_unknowns': risky})
    st = state or ('verified_success' if y == 1 else 'verified_failure' if y == 0 else 'unverified')
    return {'schema': vl.ROW_SCHEMA, 'table_version': vl.TABLE_VERSION, 'turn_key': f't{i}', 'group': group,
            'has_opportunity': True, 'gate': gate, 'features': f,
            'observed': {'post_decision': True, 'action': observed},
            'label': {'present': True, 'state': st, 'y_success': y, 'resolved': y is not None}}


def synthetic(n=400, groups=20, seed=1, signal=True):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        risky = int(rng.random() < 0.2)
        p_fail = (0.6 if risky else 0.02) if signal else 0.1
        rows.append(row(i, f'g{i % groups:02d}', 0 if rng.random() < p_fail else 1, risky=risky))
    return rows


def synthetic_graded(n, groups, seed, signal=True):
    # a fine-grained risk score (100 levels) so matched coverage is reachable without ties
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        u = rng.randrange(100)
        p_fail = 0.5 * (u / 99) ** 3 if signal else 0.1
        r = row(i, f'g{i % groups:02d}', 0 if rng.random() < p_fail else 1)
        r['features']['n_unknowns'] = u
        rows.append(r)
    return rows


class TestPieces(unittest.TestCase):
    def test_analysis_set_excludes_unverified_ask_and_missing(self):
        rows = [row(0, 'a', 1), row(1, 'a', None), row(2, 'a', 0, observed='ASK'),
                {**row(3, 'a', 1), 'has_opportunity': False, 'features': None}]
        keep, why = vl.analysis_set(rows)
        self.assertEqual([r['turn_key'] for r in keep], ['t0'])
        self.assertEqual(why, {'unverified_or_no_label': 1, 'observed_not_act': 1, 'no_opportunity': 1})

    def test_matched_coverage_never_below_target(self):
        p = [0.9, 0.8, 0.8, 0.1, 0.5]
        legal = [True] * 5
        tau = vl.tau_matched_coverage(p, legal, 0.6)
        self.assertGreaterEqual(sum(pi >= tau for pi in p) / 5, 0.6)
        self.assertEqual(vl.tau_matched_coverage(p, legal, 0.0), math.inf)

    def test_crc_threshold_controls_joint_risk(self):
        rng = random.Random(0)
        p = [rng.random() for _ in range(500)]
        y = [int(rng.random() < pi) for pi in p]
        tau = vl.tau_crc(p, [True] * 500, y, alpha=0.05)
        loss = sum(1 for pi, yi in zip(p, y) if pi >= tau and yi == 0)
        self.assertLessEqual((loss + 1) / 501, 0.05)
        self.assertEqual(vl.tau_crc([0.5], [True], [0], alpha=0.02), math.inf)  # n too small: never ACT

    def test_logistic_learns_a_signal_and_is_deterministic(self):
        rows = synthetic(200)
        X = vl.matrix(rows, FEATS)
        y = [r['label']['y_success'] for r in rows]
        p1 = vl.Logistic(steps=300).fit(X, y).predict(X)
        p2 = vl.Logistic(steps=300).fit(X, y).predict(X)
        self.assertEqual(p1, p2)
        self.assertGreater(vl.auroc(p1, y), 0.8)

    def test_group_folds_keep_groups_whole(self):
        k, fold = vl.group_folds(['b', 'a', 'c', 'a'], 5)
        self.assertEqual(k, 3)
        self.assertEqual(len(set(fold.values())), 3)

    def test_required_n(self):
        self.assertIsNone(vl.required_n_halving(0.0))
        self.assertGreater(vl.required_n_halving(0.04), vl.required_n_halving(0.2))


class TestDecision(unittest.TestCase):
    def test_tiny_table_is_insufficient_data(self):
        res = vl.evaluate(synthetic(30, groups=3), steps=100, boot_b=50)
        self.assertEqual(res['decision'], vl.INSUFFICIENT)
        self.assertTrue(res['descriptive_only'])

    def test_empty_table_is_insufficient(self):
        self.assertEqual(vl.evaluate([], steps=10, boot_b=10)['decision'], vl.INSUFFICIENT)

    def test_learned_policy_never_acts_when_act_is_illegal(self):
        rows = synthetic(120, groups=12)
        for r in rows[::7]:
            r['features']['legal.ACT'] = 0
            r['features']['legal.ASK'] = 1
        res = vl.evaluate(rows, steps=100, boot_b=50)
        self.assertEqual(res['pooled']['illegal_act_rows'], 0)

    def test_real_signal_the_gate_ignores_becomes_shadow_candidate(self):
        # gate ACTs on all but a random 20%; risk lives in a feature the gate does not use
        rows = synthetic_graded(600, groups=30, seed=3)
        rng = random.Random(9)
        for r in rows:
            if rng.random() < 0.2:
                r['gate'] = 'ASK'
                r['features']['gate=ACT'], r['features']['gate=ASK'] = 0, 1
        res = vl.evaluate(rows, steps=300, boot_b=200)
        self.assertTrue(res['sufficiency']['passed'])
        self.assertLess(res['pooled']['delta_L_cov_minus_G'], 0)
        self.assertEqual(res['decision'], vl.SHADOW, res['criteria'])

    def test_tied_scores_that_overshoot_coverage_are_not_promoted(self):
        # pre-registered: a coverage mismatch > 0.02 is NO_IMPROVEMENT even if L_cov looks better
        rows = synthetic(600, groups=30, seed=3)
        rng = random.Random(9)
        for r in rows:
            if rng.random() < 0.2:
                r['gate'] = 'ASK'
                r['features']['gate=ACT'], r['features']['gate=ASK'] = 0, 1
        res = vl.evaluate(rows, steps=300, boot_b=100)
        self.assertFalse(res['criteria']['coverage_matched'])
        self.assertEqual(res['decision'], vl.NO_IMPROVEMENT)

    def test_no_signal_is_not_promoted(self):
        rows = synthetic_graded(600, groups=30, seed=4, signal=False)
        rng = random.Random(5)
        for r in rows:
            if rng.random() < 0.2:
                r['gate'] = 'ASK'
                r['features']['gate=ACT'], r['features']['gate=ASK'] = 0, 1
        res = vl.evaluate(rows, steps=300, boot_b=200)
        self.assertNotEqual(res['decision'], vl.SHADOW)


class TestEndToEnd(unittest.TestCase):
    def test_run_writes_counts_only_report_with_projection(self):
        rows = synthetic(40, groups=4) + [row(99, 'g99', None)]
        man = {'feature_schema_sha': 'abc', 'counts': {},
               'sweep': {'first_opportunity_at': '2026-09-30T00:00:00Z',
                         'total': {'turns': 430, 'verified_success': 100, 'verified_failure': 4, 'contested': 1,
                                   'with_opportunity': 20, 'with_opportunity_and_resolved': 7},
                         'by_day': {'2026-09-29': {'turns': 60}, '2026-09-30': {'turns': 80, 'with_opportunity': 20}}}}
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / 't.jsonl'
            t.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            t.with_suffix('.manifest.json').write_text(json.dumps(man))
            out = Path(d) / 'r.json'
            md = Path(d) / 'r.md'
            self.assertEqual(vl.main(['--table', str(t), '--out', str(out), '--md', str(md)]), 0)
            res = json.loads(out.read_text())
            self.assertEqual(res['primary']['decision'], vl.INSUFFICIENT)
            sc = res['projection']['scenarios']
            self.assertAlmostEqual(sc['current_feature_coverage']['feature_coverage'], 0.25)
            self.assertGreater(sc['current_feature_coverage']['weeks_to_sufficiency_gate'],
                               sc['full_feature_coverage']['weeks_to_sufficiency_gate'])
            self.assertNotIn('t0', out.read_text())
            self.assertIn('INSUFFICIENT_DATA', md.read_text())

    def test_wrong_table_version_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / 't.jsonl'
            t.write_text(json.dumps({**row(0, 'a', 1), 'table_version': '9.9'}) + '\n')
            with self.assertRaises(ValueError):
                vl.load_table(t)


if __name__ == '__main__':
    unittest.main()


class TestSpan(unittest.TestCase):
    def test_short_history_is_not_divided_by_seven(self):
        sw = {'first_turn_at': '2026-09-30T00:00:00Z', 'measured_at': '2026-10-01T12:00:00Z'}
        self.assertAlmostEqual(vl.observed_span_days(sw), 1.5)
        self.assertEqual(vl.observed_span_days({}), 7.0)
        self.assertEqual(vl.observed_span_days({'first_turn_at': '2026-01-01T00:00:00Z',
                                                'measured_at': '2026-10-01T00:00:00Z'}), 7.0)
