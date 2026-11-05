"""Regression tests for the read-only ABAB decision rules."""

from __future__ import annotations

import unittest

from evolution_lab.abab_state import (
    Experiment,
    Hypothesis,
    World,
    analog_ok,
    discriminate,
    experiment_dominates,
    next_a,
    priority,
    rank_open,
    should_slow,
    source_weight,
    stage_c_ready,
    stop_reason,
)


class AbabStateRulesTests(unittest.TestCase):
    def test_priority(self):
        self.assertAlmostEqual(priority(Hypothesis("a", "c")), 0.0375)
        self.assertAlmostEqual(priority(Hypothesis("a", "c", cost=0)), 37500.0)

    def test_rank_open_filters_and_sorts(self):
        low = Hypothesis("low", "low claim", eig=0.1)
        middle = Hypothesis("middle", "middle claim")
        high = Hypothesis("high", "high claim", eig=0.9, status="rival")
        excluded = [
            Hypothesis(status, "excluded", eig=1.0, status=status)
            for status in ("pruned", "promoted", "deferred")
        ]
        world = World("o", hypotheses=[low, *excluded, high, middle])
        self.assertEqual(
            [hypothesis.id for hypothesis in rank_open(world)],
            ["high", "middle", "low"],
        )
        self.assertEqual(rank_open(World("o", hypotheses=excluded)), [])

    def test_source_weight(self):
        expected = {
            "measurement": 1.0,
            "primary": 0.85,
            "official": 0.8,
            "analysis": 0.55,
            "secondary": 0.25,
            "unknown": 0.2,
        }
        for source_class, weight in expected.items():
            with self.subTest(source_class=source_class):
                self.assertAlmostEqual(source_weight(source_class), weight)

    def test_should_slow(self):
        self.assertFalse(should_slow(World("o"), None))
        self.assertFalse(should_slow(World("o"), "unrecognized"))
        for event in (
            "contradiction",
            "plateau",
            "surprise",
            "graph_change",
            "repeat_fail",
        ):
            with self.subTest(event=event):
                self.assertTrue(should_slow(World("o"), event))

        self.assertFalse(should_slow(World("o", no_update_streak=1)))
        self.assertTrue(should_slow(World("o", no_update_streak=2)))

        first = Experiment(
            "e1", "a", "intervention", "baseline", "prediction", "metric",
            outcome="revert",
        )
        second = Experiment(
            "e2", "a", "intervention", "baseline", "prediction", "metric",
            outcome="revert",
        )
        self.assertFalse(should_slow(World("o", experiments=[first])))
        self.assertTrue(should_slow(World("o", experiments=[first, second])))

    def test_experiment_dominates(self):
        self.assertFalse(experiment_dominates(World("o")))
        falsifiable = Hypothesis("a", "c", falsifiers=["f"], cost=2.0, eig=0.3)
        self.assertTrue(
            experiment_dominates(World("o", hypotheses=[falsifiable]))
        )

        costly_mechanism = Hypothesis("a", "c", mechanism="m", cost=1.5)
        self.assertFalse(
            experiment_dominates(
                World("o", hypotheses=[costly_mechanism], wave=3)
            )
        )
        self.assertTrue(
            experiment_dominates(
                World("o", hypotheses=[costly_mechanism], wave=4)
            )
        )

    def test_stop_reason_each_branch(self):
        self.assertEqual(stop_reason(World("o", wave=8)), "budget_exhausted")
        self.assertEqual(
            stop_reason(World("o", wave=8, no_update_streak=3)),
            "budget_exhausted",
        )
        self.assertEqual(
            stop_reason(World("o", no_update_streak=3)),
            "repeated_no_update",
        )
        self.assertEqual(stop_reason(World("o", wave=1)), "objective_resolved")
        self.assertIsNone(stop_reason(World("o")))

        falsifiable = Hypothesis("a", "c", falsifiers=["f"], cost=2.0, eig=0.3)
        self.assertEqual(
            stop_reason(World("o", hypotheses=[falsifiable], wave=3)),
            "experiment_dominates",
        )
        self.assertEqual(
            stop_reason(
                World(
                    "o",
                    hypotheses=[Hypothesis("a", "c", decision_change=0.01)],
                    wave=2,
                )
            ),
            "cannot_change_decision",
        )
        self.assertEqual(
            stop_reason(
                World(
                    "o",
                    hypotheses=[Hypothesis("a", "c", transferability=0.1)],
                    wave=2,
                )
            ),
            "eig_collapsed",
        )
        self.assertIsNone(
            stop_reason(World("o", hypotheses=[Hypothesis("a", "c")], wave=2))
        )

    def test_stage_c_ready(self):
        self.assertFalse(stage_c_ready(World("o")))
        self.assertFalse(
            stage_c_ready(World("o", hypotheses=[Hypothesis("a", "c")]))
        )
        self.assertFalse(
            stage_c_ready(
                World(
                    "o",
                    hypotheses=[
                        Hypothesis("a", "c", mechanism="m", status="pruned")
                    ],
                )
            )
        )
        self.assertTrue(
            stage_c_ready(
                World(
                    "o",
                    hypotheses=[
                        Hypothesis("a", "c", mechanism="m", status="promoted")
                    ],
                )
            )
        )
        self.assertTrue(
            stage_c_ready(
                World(
                    "o",
                    hypotheses=[Hypothesis("a", "c", mechanism="m", cost=1.5)],
                    wave=4,
                )
            )
        )

    def test_analog_ok(self):
        self.assertTrue(analog_ok("m", "M ", True))
        self.assertFalse(analog_ok("m", "m", False))
        self.assertFalse(analog_ok("", "m", True))
        # Mapped analogs with distinct mechanism strings are currently accepted.
        self.assertTrue(analog_ok("a", "b", True))

    def test_next_a_modes(self):
        empty_strategy = next_a(World("o"))
        self.assertEqual(empty_strategy.stop, "objective_resolved")
        self.assertEqual(empty_strategy.target_uncertainty, "none")

        for wave, mode in (
            (0, "discovery"),
            (5, "mechanism"),
            (10, "falsification"),
            (15, "architecture"),
            (20, "critical_path"),
        ):
            with self.subTest(wave=wave):
                strategy = next_a(
                    World("o", hypotheses=[Hypothesis("a", "c")], deep=True,
                          wave=wave)
                )
                self.assertEqual(strategy.mode, mode)

        self.assertEqual(
            next_a(
                World("o", hypotheses=[Hypothesis("a", "c", rival_of="b")])
            ).mode,
            "falsification",
        )
        self.assertEqual(
            next_a(
                World(
                    "o",
                    hypotheses=[Hypothesis("a", "c")],
                    loop="slow",
                    contradictions=["conflicting result"],
                )
            ).mode,
            "mechanism",
        )
        dominating = Hypothesis("a", "c", falsifiers=["f"], cost=2.0, eig=0.3)
        self.assertEqual(
            next_a(World("o", hypotheses=[dominating])).mode,
            "implementation",
        )
        self.assertEqual(
            next_a(
                World(
                    "o",
                    hypotheses=[dominating],
                    loop="slow",
                    contradictions=["conflicting result"],
                )
            ).mode,
            "mechanism",
        )

        with_falsifiers = next_a(
            World(
                "o",
                hypotheses=[Hypothesis("a", "c", falsifiers=["f1", "f2"])],
            )
        )
        self.assertEqual(with_falsifiers.evidence_needed, ["f1", "f2"])
        self.assertEqual(with_falsifiers.falsifier, "f1")
        self.assertEqual(with_falsifiers.counterexample, "f1")

        no_falsifiers = next_a(
            World("o", hypotheses=[Hypothesis("a", "c")])
        )
        self.assertEqual(
            no_falsifiers.evidence_needed, ["discriminating measurement"]
        )
        self.assertEqual(no_falsifiers.falsifier, "")
        self.assertEqual(
            no_falsifiers.source_classes,
            ["measurement", "primary", "official"],
        )

    def test_discriminate(self):
        self.assertEqual(
            discriminate(
                Hypothesis("A", "c", falsifiers=["x", "y"]),
                Hypothesis("B", "c", falsifiers=["y"]),
            ),
            "x",
        )
        self.assertEqual(
            discriminate(
                Hypothesis("A", "c", falsifiers=["y"]),
                Hypothesis("B", "c", falsifiers=["y", "z"]),
            ),
            "y",
        )
        self.assertEqual(
            discriminate(Hypothesis("A", "c"), Hypothesis("B", "c")),
            "measure:A-vs-B",
        )


if __name__ == "__main__":
    unittest.main()
