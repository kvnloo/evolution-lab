import unittest

import numpy as np

from evolution_lab.real_recovery import (
    LOOSE_MAP,
    RAW_ACTIONS,
    STRICT_MAP,
    classify_result,
    error_class,
    gym_frame,
    history_episodes,
    is_restart_call,
    label_of,
    n_rich_features,
    rich_features,
)
from evolution_lab.schema import ACTIONS
from evolution_lab.task import n_features, teacher_action


def _ep(**kw):
    base = {
        "lineage": "L",
        "source": "cron",
        "tool_family": "terminal",
        "err": {"transient": False, "sandbox_dead": False, "policy": False, "hard": True, "unfamiliar": False, "benign": False},
        "exit_bucket": "1",
        "consec_fail_same_tool": 0,
        "session_fail_count": 0,
        "turn_idx": 3,
        "n_parallel": 1,
        "n_failed_in_turn": 1,
        "last_action": "noop",
        "raw_action": "retry_exact",
        "local_recovered": True,
        "session_complete": True,
        "ts": 1.0,
        "extra": {},
    }
    base.update(kw)
    return base


class RealRecoveryTest(unittest.TestCase):
    def test_maps_cover_raw_actions_and_targets_are_actions(self):
        for m in (STRICT_MAP, LOOSE_MAP):
            self.assertEqual(set(m), set(RAW_ACTIONS))
            for v in m.values():
                self.assertTrue(v is None or v in ACTIONS)

    def test_label_requires_verified_good_outcome(self):
        self.assertEqual(label_of(_ep(), "strict"), ACTIONS.index("retry"))
        self.assertEqual(label_of(_ep(local_recovered=False), "strict"), -1)
        self.assertEqual(label_of(_ep(local_recovered=None), "loose"), -1)
        self.assertEqual(label_of(_ep(session_complete=False), "loose"), -1)
        self.assertEqual(label_of(_ep(raw_action="switch_tool"), "strict"), -1)
        self.assertEqual(label_of(_ep(raw_action="switch_tool"), "loose"), ACTIONS.index("noop"))
        self.assertEqual(label_of(_ep(raw_action="switch_tool"), "loose_esc"), ACTIONS.index("escalate"))

    def test_structured_failure_detection(self):
        ok = classify_result("terminal", '{"output": "x"}', None, 0, None, None, None, None, None, "x")
        self.assertFalse(ok.failed)
        bad = classify_result("terminal", "{}", None, 2, None, None, None, None, None, "No such file")
        self.assertTrue(bad.failed)
        pol = classify_result("terminal", "{}", None, None, None, None, None, True, None, "")
        self.assertTrue(pol.failed and error_class(pol).policy)
        txt = classify_result("mcp__x", "Error: 503 Service Unavailable", None, None, None, None, None, None, None, None)
        self.assertTrue(txt.failed and error_class(txt).transient)
        benign = classify_result("terminal", "{}", None, 1, None, None, "no matches", None, None, "")
        self.assertTrue(error_class(benign).benign)

    def test_restart_detection(self):
        self.assertTrue(is_restart_call("terminal", '{"command": "systemctl --user restart foo"}'))
        self.assertTrue(is_restart_call("process", {"action": "kill"}))
        self.assertFalse(is_restart_call("terminal", '{"command": "ls -la"}'))

    def test_gym_frame_layout_drives_rule(self):
        f = gym_frame(_ep(err={**_ep()["err"], "policy": True}))
        self.assertEqual(f.shape, (n_features(),))
        self.assertEqual(teacher_action(f), ACTIONS.index("page_human"))
        f = gym_frame(_ep(err={**_ep()["err"], "sandbox_dead": True}))
        self.assertEqual(teacher_action(f), ACTIONS.index("restart_sandbox"))
        self.assertEqual(rich_features(_ep()).shape, (n_rich_features(),))

    def test_history_is_lineage_local_and_time_ordered(self):
        eps = [_ep(ts=2.0, turn_idx=5), _ep(ts=1.0, turn_idx=1), _ep(lineage="M", ts=0.5)]
        H = history_episodes(eps, rich=False)
        self.assertEqual(H[0].frames.shape, (8, n_features()))
        # episode 0 (later) sees episode 1's frame just before its own
        np.testing.assert_allclose(H[0].frames[-2], gym_frame(eps[1]))
        np.testing.assert_allclose(H[0].frames[-1], gym_frame(eps[0]))
        # other lineage is padded with its own frame only
        np.testing.assert_allclose(H[2].frames[0], gym_frame(eps[2]))


if __name__ == "__main__":
    unittest.main()
