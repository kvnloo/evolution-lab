from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from evolution_lab.abab_state import (
    Evidence,
    Experiment,
    Hypothesis,
    Strategy,
    World,
    apply_mutation,
    learn_payload,
    load,
    preserve_rivals,
    record_experiment_outcome,
    save,
    wave_heartbeat,
)


def _experiment(exp_id: str = "e", hypothesis_id: str = "a", **kwargs) -> Experiment:
    return Experiment(
        id=exp_id,
        hypothesis_id=hypothesis_id,
        intervention="toggle",
        baseline="baseline",
        prediction="improve",
        metric="score",
        **kwargs,
    )


class AbabStateMutationsTests(unittest.TestCase):
    def test_apply_mutation_does_not_mutate_input(self):
        world = World(
            "o",
            hypotheses=[Hypothesis("a", "c"), Hypothesis("b", "d")],
            evidence=[Evidence("ev", "claim", "source", "measurement")],
            experiments=[_experiment()],
            concepts={"prior": "value"},
        )
        before = deepcopy(world)
        cases = [
            ("ADD", {"hypothesis": {"id": "c", "claim": "new"}}),
            ("MERGE", {"keep": "a", "drop": "b"}),
            ("SPLIT", {"parent": "a", "children": [{"id": "c", "claim": "new"}]}),
            ("UPGRADE", {"id": "a"}),
            ("DOWNGRADE", {"id": "a"}),
            ("PRUNE", {"id": "a"}),
            ("DEFER", {"id": "a"}),
            ("CONTRADICT", {"id": "a", "note": "counterexample"}),
            ("SPAWN_EXPERIMENT", {"experiment": asdict(_experiment("new"))}),
            ("SPAWN_RESEARCH_BRANCH", {"hypothesis": {"id": "c", "claim": "new"}}),
            ("NO_UPDATE", {"note": "n"}),
        ]
        for mutation, payload in cases:
            with self.subTest(mutation=mutation):
                changed = apply_mutation(world, mutation, payload)
                self.assertIsNot(changed, world)
                self.assertEqual(changed.wave, 1)
                self.assertEqual(world, before)
                self.assertEqual(world.wave, 0)
                self.assertEqual(world.hypotheses, before.hypotheses)
                self.assertIsNot(changed.hypotheses, world.hypotheses)
                self.assertIsNot(changed.hypotheses[0], world.hypotheses[0])

    def test_no_update_streak_stops_world(self):
        world = World("o", hypotheses=[Hypothesis("a", "c")])
        for wave in (1, 2, 3):
            world = apply_mutation(world, "NO_UPDATE", {"note": "n"})
            self.assertEqual(world.wave, wave)
            self.assertEqual(world.no_update_streak, wave)
        self.assertTrue(world.stopped)
        self.assertEqual(world.stop_reason, "repeated_no_update")
        self.assertEqual(
            world.strategy_log[0],
            {"wave": 1, "mutation": "NO_UPDATE", "note": "n"},
        )
        self.assertEqual(len(world.strategy_log), 3)

    def test_add_variants(self):
        base = World(
            "o", hypotheses=[Hypothesis("a", "c")], no_update_streak=2
        )
        with_hypothesis = apply_mutation(
            base, "ADD", {"hypothesis": {"id": "b", "claim": "d"}}
        )
        self.assertEqual([h.id for h in with_hypothesis.hypotheses], ["a", "b"])
        self.assertIsInstance(with_hypothesis.hypotheses[-1], Hypothesis)
        self.assertEqual(with_hypothesis.hypotheses[-1].claim, "d")

        with_evidence = apply_mutation(
            base,
            "ADD",
            {"evidence": {
                "id": "ev", "claim": "measured", "source": "local",
                "source_class": "measurement",
            }},
        )
        self.assertEqual(len(with_evidence.evidence), 1)
        self.assertIsInstance(with_evidence.evidence[0], Evidence)
        self.assertEqual(with_evidence.evidence[0].claim, "measured")

        with_concept = apply_mutation(
            base, "ADD", {"concept": ("k", "v")}
        )
        self.assertEqual(with_concept.concepts["k"], "v")

        # Already-instantiated values are accepted as well as dictionaries.
        with_objects = apply_mutation(
            base,
            "ADD",
            {
                "hypothesis": Hypothesis("b", "d"),
                "evidence": Evidence("ev", "measured", "local", "measurement"),
            },
        )
        self.assertIsInstance(with_objects.hypotheses[-1], Hypothesis)
        self.assertIsInstance(with_objects.evidence[-1], Evidence)

        for result in (
            with_hypothesis, with_evidence, with_concept, with_objects
        ):
            self.assertEqual(result.wave, 1)
            self.assertEqual(result.no_update_streak, 0)
            self.assertEqual(
                result.strategy_log[-1], {"wave": 1, "mutation": "ADD"}
            )
        self.assertEqual(base.no_update_streak, 2)
        self.assertEqual(len(base.hypotheses), 1)

    def test_merge_split_upgrade_downgrade_prune_defer(self):
        merged = apply_mutation(
            World("o", hypotheses=[
                Hypothesis("a", "c", status="rival"),
                Hypothesis("b", "d"),
            ]),
            "MERGE",
            {"keep": "a", "drop": "b"},
        )
        self.assertEqual([h.id for h in merged.hypotheses], ["a"])
        self.assertEqual(merged.hypotheses[0].status, "open")

        split = apply_mutation(
            World("o", hypotheses=[Hypothesis("a", "c")]),
            "SPLIT",
            {"parent": "a", "children": [{"id": "a1", "claim": "x"}]},
        )
        self.assertEqual([h.id for h in split.hypotheses], ["a", "a1"])
        self.assertEqual(split.hypotheses[0].status, "deferred")
        self.assertIsInstance(split.hypotheses[1], Hypothesis)
        self.assertEqual(split.hypotheses[1].claim, "x")

        upgraded = apply_mutation(
            World("o", hypotheses=[Hypothesis("a", "c", eig=0.95)]),
            "UPGRADE",
            {"id": "a"},
        )
        self.assertEqual(upgraded.hypotheses[0].status, "promoted")
        self.assertEqual(upgraded.hypotheses[0].eig, 1.0)

        downgraded = apply_mutation(
            World("o", hypotheses=[Hypothesis("a", "c", eig=0.1)]),
            "DOWNGRADE",
            {"id": "a"},
        )
        self.assertEqual(downgraded.hypotheses[0].eig, 0.0)

        pruned = apply_mutation(
            World("o", hypotheses=[Hypothesis("a", "c")]),
            "PRUNE",
            {"id": "a"},
        )
        self.assertEqual(pruned.hypotheses[0].status, "pruned")
        self.assertEqual(pruned.dead_ends, ["c"])

        deferred = apply_mutation(
            World("o", hypotheses=[Hypothesis("a", "c")]),
            "DEFER",
            {"id": "a"},
        )
        self.assertEqual(deferred.hypotheses[0].status, "deferred")

    def test_contradict_and_spawn(self):
        base = World("o", hypotheses=[Hypothesis("a", "c")])
        contradicted = apply_mutation(
            base, "CONTRADICT", {"note": "n", "id": "a"}
        )
        self.assertEqual(contradicted.contradictions, ["n"])
        self.assertEqual(contradicted.loop, "slow")
        self.assertEqual(contradicted.hypotheses[0].status, "rival")

        spawned_experiment = apply_mutation(
            base,
            "SPAWN_EXPERIMENT",
            {"experiment": asdict(_experiment("e1"))},
        )
        self.assertEqual(len(spawned_experiment.experiments), 1)
        self.assertIsInstance(spawned_experiment.experiments[0], Experiment)
        self.assertEqual(spawned_experiment.experiments[0].id, "e1")
        self.assertEqual(
            spawned_experiment.experiments[0].hypothesis_id, "a"
        )

        spawned_branch = apply_mutation(
            base,
            "SPAWN_RESEARCH_BRANCH",
            {"hypothesis": {"id": "b", "claim": "d"}},
        )
        self.assertEqual([h.id for h in spawned_branch.hypotheses], ["a", "b"])
        self.assertIsInstance(spawned_branch.hypotheses[1], Hypothesis)
        self.assertEqual(spawned_branch.hypotheses[1].claim, "d")

    def test_event_payload_forces_slow_loop(self):
        world = World("o", hypotheses=[Hypothesis("a", "c")])
        changed = apply_mutation(
            world, "ADD", {"concept": ("k", "v"), "event": "surprise"}
        )
        self.assertEqual(changed.concepts["k"], "v")
        self.assertEqual(changed.loop, "slow")
        self.assertEqual(world.loop, "fast")

    def test_record_experiment_outcome(self):
        base = World(
            "o",
            hypotheses=[Hypothesis("a", "c", eig=0.6, status="deferred")],
            experiments=[_experiment()],
            no_update_streak=2,
            wave=2,
        )
        kept = record_experiment_outcome(
            base, "e", "keep", {"score": 1.0}
        )
        self.assertEqual(kept.experiments[0].outcome, "keep")
        self.assertEqual(kept.experiments[0].measured, {"score": 1.0})
        self.assertEqual(kept.hypotheses[0].status, "promoted")
        self.assertEqual(kept.wave, 3)
        self.assertEqual(kept.no_update_streak, 0)
        self.assertEqual(base.experiments[0].outcome, None)
        self.assertEqual(base.hypotheses[0].status, "deferred")

        reverted = record_experiment_outcome(
            base, "e", "revert", {"score": 0.0}
        )
        self.assertEqual(reverted.experiments[0].outcome, "revert")
        self.assertEqual(reverted.experiments[0].measured, {"score": 0.0})
        self.assertEqual(reverted.hypotheses[0].status, "open")
        self.assertEqual(reverted.hypotheses[0].eig, 0.75)
        self.assertEqual(reverted.dead_ends, ["experiment:e"])
        self.assertEqual(reverted.loop, "slow")
        self.assertEqual(reverted.wave, 3)
        self.assertEqual(reverted.no_update_streak, 0)

        capped = record_experiment_outcome(
            World(
                "o",
                hypotheses=[Hypothesis("a", "c", eig=0.95)],
                experiments=[_experiment()],
            ),
            "e", "revert", {},
        )
        self.assertEqual(capped.hypotheses[0].eig, 1.0)

        unknown_base = World(
            "o",
            hypotheses=[Hypothesis("a", "c")],
            experiments=[_experiment()],
            concepts={"existing": "value"},
            dead_ends=["prior"],
            loop="slow",
            no_update_streak=2,
            wave=4,
        )
        expected = deepcopy(unknown_base)
        expected.wave += 1
        expected.no_update_streak = 0
        unknown = record_experiment_outcome(
            unknown_base, "missing", "keep", {"score": 2}
        )
        self.assertEqual(unknown, expected)
        self.assertEqual(unknown_base.wave, 4)
        self.assertEqual(unknown_base.no_update_streak, 2)

    def test_preserve_rivals(self):
        world = World(
            "o", hypotheses=[Hypothesis("a", "c"), Hypothesis("b", "d")]
        )
        changed = preserve_rivals(world, "a", "b")
        rivals = {h.id: h for h in changed.hypotheses}
        self.assertEqual(rivals["a"].status, "rival")
        self.assertEqual(rivals["b"].status, "rival")
        self.assertEqual(rivals["a"].rival_of, "b")
        self.assertEqual(rivals["b"].rival_of, "a")
        self.assertEqual([h.status for h in world.hypotheses], ["open", "open"])
        self.assertEqual([h.rival_of for h in world.hypotheses], [None, None])

    def test_learn_payload(self):
        kept = _experiment("kept", outcome="keep", measured={"score": 1.0})
        reverted = _experiment("failed", outcome="revert", measured={"score": 0.0})
        world = World(
            "o",
            hypotheses=[Hypothesis("a", "c", status="promoted")],
            experiments=[kept, reverted],
            dead_ends=["experiment:failed"],
        )
        payload = learn_payload(world)
        self.assertEqual(payload["objective"], "o")
        self.assertEqual(payload["validated_experiments"], [asdict(kept)])
        self.assertEqual(payload["failed_experiments"], [asdict(reverted)])
        self.assertIsInstance(payload["validated_experiments"][0], dict)
        self.assertIsInstance(payload["failed_experiments"][0], dict)
        self.assertEqual(payload["dead_ends"], ["experiment:failed"])
        self.assertEqual(payload["promoted"], [asdict(world.hypotheses[0])])
        self.assertIs(payload["require_task_validation_before_default"], True)
        self.assertIs(payload["skip_if_no_measurement"], False)

        no_keeps = learn_payload(World("o", experiments=[reverted]))
        self.assertEqual(no_keeps["validated_experiments"], [])
        self.assertIs(no_keeps["skip_if_no_measurement"], True)
        self.assertIs(
            learn_payload(World("o"))["skip_if_no_measurement"], True
        )

    def test_save_load_roundtrip(self):
        world = World(
            "o",
            constraints=["bounded"],
            metrics=["score"],
            baseline="baseline",
            concepts={"k": "v"},
            mechanisms={"best": "mechanism"},
            hypotheses=[Hypothesis("a", "c", predictions=["p"], eig=0.6)],
            evidence=[Evidence(
                "ev", "measured", "local", "measurement", confidence=0.9
            )],
            experiments=[_experiment(outcome="keep", measured={"score": 0.7})],
            strategy_log=[{"wave": 1, "mutation": "ADD"}],
            dead_ends=["prior"],
            wave=1,
            last_strategy=Strategy("t", "a", "r", "m", neighbors=["neighbor"]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x" / "abab.json"
            self.assertFalse(path.parent.exists())
            save(world, path)
            self.assertTrue(path.is_file())
            loaded = load(path)

        self.assertIsInstance(loaded, World)
        self.assertIsInstance(loaded.hypotheses[0], Hypothesis)
        self.assertIsInstance(loaded.evidence[0], Evidence)
        self.assertIsInstance(loaded.experiments[0], Experiment)
        self.assertEqual(loaded.hypotheses, world.hypotheses)
        self.assertEqual(loaded.evidence, world.evidence)
        self.assertEqual(loaded.experiments, world.experiments)
        self.assertIsNone(loaded.last_strategy)  # load intentionally drops it.
        expected = deepcopy(world)
        expected.last_strategy = None
        self.assertEqual(loaded, expected)

    def test_wave_heartbeat(self):
        empty = wave_heartbeat(World("o"), "change", "why", "failed", "next")
        self.assertEqual(empty, {
            "CURRENT_MODEL": "o",
            "WHAT_CHANGED": "change",
            "WHY": "why",
            "WHAT_FAILED": "failed",
            "GRAPH_DELTA": "none",
            "HIGHEST_VALUE_UNCERTAINTY": "none",
            "NEXT_MOVE": "next",
        })

        with_baseline = wave_heartbeat(
            World("o", baseline="b"), "", "", "", ""
        )
        self.assertEqual(with_baseline["CURRENT_MODEL"], "b")

        with_best = wave_heartbeat(
            World("o", baseline="b", mechanisms={"best": "m"}),
            "", "", "", "",
        )
        self.assertEqual(with_best["CURRENT_MODEL"], "m")

        changed = apply_mutation(
            World("o"), "ADD", {"hypothesis": {"id": "a", "claim": "c"}}
        )
        heartbeat = wave_heartbeat(changed, "", "", "", "")
        self.assertEqual(heartbeat["GRAPH_DELTA"], "ADD")
        self.assertEqual(heartbeat["HIGHEST_VALUE_UNCERTAINTY"], "c")


if __name__ == "__main__":
    unittest.main()
