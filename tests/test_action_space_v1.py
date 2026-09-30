import unittest

import numpy as np

from evolution_lab.action_space_v1 import (
    A1,
    ACTIONS_V1,
    COST,
    IDX,
    label_v1,
    min_cost_decode,
    overlap,
    plasticity_train_k,
    recovered,
    rule_v1,
    salient_tokens,
    taken_action,
    v1_frame,
)


def call(**kw):
    c = {"fam": "terminal", "same_tool": False, "exact": False, "jac": 0.0, "ovl": 0.0, "path_shared": 0,
         "restart": False, "clarify": False, "delegate": False, "ok": True}
    c.update(kw)
    return c


def turn(calls=(), users=0, fail_report=None):
    return {"users_before": users, "fail_report": fail_report, "calls": list(calls)}


class TestTakenAction(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(taken_action([], interactive=False), (None, "R0_none"))
        self.assertIsNone(taken_action([turn([call()], users=1)], interactive=True)[0])
        self.assertEqual(taken_action([turn()], interactive=True)[0], "ask")
        self.assertEqual(taken_action([turn(fail_report=True)], interactive=False)[0], "abort")
        self.assertEqual(taken_action([turn(fail_report=False)], interactive=False)[0], "noop")
        self.assertEqual(taken_action([turn([call(clarify=True), call(same_tool=True, exact=True)])], interactive=True)[0], "ask")
        self.assertEqual(taken_action([turn([call(delegate=True)])], interactive=False)[0], "switch_tool")
        self.assertEqual(taken_action([turn([call(restart=True)])], interactive=False)[0], "edit_retry")
        self.assertEqual(taken_action([turn([call(same_tool=True, exact=True, jac=1.0)])], interactive=False)[0], "retry")
        self.assertEqual(taken_action([turn([call(same_tool=True, jac=0.6)])], interactive=False)[0], "edit_retry")
        self.assertEqual(taken_action([turn([call(same_tool=True, ovl=0.3)])], interactive=False),
                         ("edit_retry", "R7_same_tool_same_target"))
        self.assertEqual(taken_action([turn([call(path_shared=1)])], interactive=False)[0], "switch_tool")
        self.assertEqual(taken_action([turn([call(same_tool=True, jac=0.1)])], interactive=False),
                         ("noop", "R9_moved_on"))


class TestRecovered(unittest.TestCase):
    def test_retry_needs_success_within_3(self):
        sk = [turn([call(same_tool=True, exact=True, ok=False)]), turn([call(same_tool=True, jac=0.7, ok=True)])]
        self.assertTrue(recovered("retry", "R5_exact", sk, True))
        self.assertFalse(recovered("retry", "R5_exact", sk, False))
        sk = [turn([call(same_tool=True, exact=True, ok=False)])]
        self.assertFalse(recovered("retry", "R5_exact", sk, True))

    def test_noop_invalidated_by_failed_reattempt(self):
        sk = [turn([call()]), turn([call(same_tool=True, ovl=0.5, ok=False)])]
        self.assertFalse(recovered("noop", "R9_moved_on", sk, True))
        self.assertTrue(recovered("noop", "R9_moved_on", sk[:1], True))

    def test_ask_interactive(self):
        self.assertTrue(recovered("ask", "R1_stop_interactive", [turn(), turn([call()], users=1)], True))
        self.assertIsNone(recovered("ask", "R1_stop_interactive", [turn()], True))

    def test_label_filter(self):
        ep = {"v1_action": "abort", "v1_recovered": True}
        self.assertEqual(label_v1(ep), IDX["abort"])
        self.assertEqual(label_v1({"v1_action": "retry", "v1_recovered": False}), -1)
        self.assertEqual(label_v1({"v1_action": "retry", "v1_recovered": False}, filtered=False), IDX["retry"])
        self.assertEqual(label_v1({"v1_action": None, "v1_recovered": None}, filtered=False), -1)


class TestCost(unittest.TestCase):
    def test_structure(self):
        self.assertEqual(COST.shape, (A1, A1))
        self.assertTrue(np.all(np.diag(COST) == 0))
        ab, q, r, e = IDX["abort"], IDX["ask"], IDX["retry"], IDX["edit_retry"]
        self.assertEqual(COST[r, ab], 5.0)
        self.assertEqual(COST[r, q], 1.5)
        self.assertEqual(COST[q, r], 4.0)
        self.assertEqual(COST[e, r], 0.5)
        # wrong abort is the most expensive error in every row
        for t in range(A1):
            if t != ab:
                self.assertEqual(COST[t].max(), COST[t, ab])

    def test_min_cost_decode_prefers_cheap_hedge(self):
        P = np.zeros((1, A1))
        P[0, IDX["abort"]] = 0.55
        P[0, IDX["noop"]] = 0.45
        self.assertNotEqual(min_cost_decode(P)[0], IDX["abort"])


class TestTokensAndRule(unittest.TestCase):
    def test_salient_tokens_drop_keys(self):
        t = salient_tokens({"command": "cat /home/u/proj/main.py", "path": None, "n": 12345})
        self.assertIn("home/u/proj/main.py", t)
        self.assertNotIn("command", t)
        self.assertNotIn("12345", t)
        ovl, ps = overlap(t, salient_tokens({"path": "/home/u/proj/main.py"}))
        self.assertGreater(ovl, 0)
        self.assertEqual(ps, 1)

    def test_rule_never_aborts(self):
        base = {"source": "cron", "consec_fail_same_tool": 0,
                "err": {k: False for k in ("benign", "policy", "sandbox_dead", "transient", "unfamiliar", "hard")}}
        seen = set()
        for k in base["err"]:
            for src in ("cron", "interactive"):
                for cf in (0, 3):
                    ep = {**base, "source": src, "consec_fail_same_tool": cf, "err": {**base["err"], k: True}}
                    seen.add(ACTIONS_V1[rule_v1(ep)])
        self.assertNotIn("abort", seen)

    def test_frame_width(self):
        ep = {"err": {"transient": False, "sandbox_dead": False, "policy": False, "hard": True, "unfamiliar": False,
                      "benign": False}, "turn_idx": 3, "consec_fail_same_tool": 0, "last_action": "noop",
              "source": "cron", "tool_family": "terminal", "exit_bucket": "1", "session_fail_count": 0,
              "n_parallel": 1, "n_failed_in_turn": 1, "v1_prev_action": "switch_tool"}
        f = v1_frame(ep)
        self.assertEqual(f.shape, (10 + A1 + 6 + 10 + 6 + 4,))
        self.assertEqual(f[10 + IDX["switch_tool"]], 1.0)


class TestPlasticityParity(unittest.TestCase):
    def test_matches_reference_trainer_for_5_actions(self):
        from evolution_lab.models import _kc_codes, _pn_features, _plasticity_train, _sparse_pn_kc
        from evolution_lab.task import N_ACTIONS, episode_from_prefix

        rng = np.random.default_rng(0)
        eps = [episode_from_prefix([rng.random(15) for _ in range(3)], history=4) for _ in range(40)]
        y = rng.integers(0, N_ACTIONS, 40)
        X = _pn_features(eps)
        W_pn = _sparse_pn_kc(X.shape[1], 64, np.random.default_rng(1))
        ref = _plasticity_train(eps, y, W_pn_kc=W_pn, k_winners=6, W=np.zeros((64, N_ACTIONS)),
                                rng=np.random.default_rng(2), epochs=3, lr=0.35)
        mine = plasticity_train_k(_kc_codes(X, W_pn, 6), y, W=np.zeros((64, N_ACTIONS)),
                                  rng=np.random.default_rng(2), epochs=3, lr=0.35)
        np.testing.assert_array_equal(ref, mine)


if __name__ == "__main__":
    unittest.main()
