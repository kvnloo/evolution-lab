"""Phase 1B section I tests: episode compilation, label hierarchy and splits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab.episode import (
    Episode,
    EpisodeError,
    build_split_plan,
    compile_episode,
    compile_episodes,
    label_census,
    selective_teacher_candidates,
)


def _receipt(**overrides):
    row = {
        "schema": "z0int.phase1b.observation.v1",
        "run_id": "p1b-test",
        "trace_id": "trace-1",
        "observed_at": "2026-09-21T10:00:00Z",
        "state_id": "one_obvious_tool",
        "state_family": "tool_selection",
        "state_features": {"candidate_action_count": 3, "family": "tool_selection",
                           "legal_family_count": 2, "legal_risk_classes": ["read"],
                           "authority_breadth": 1, "budget_units": 8,
                           "declared_dangerous_count": 0, "satisfied_count": 0},
        "legal_actions": ["fs.read", "fs.list", "abstain"],
        "arm": "compiler+hammer2.1_3b",
        "arm_alias": "SUB_compiler_hammer3b",
        "model_id": "hammer2.1_3b",
        "model_revision": "abc",
        "quant": "Q4_K_M",
        "compiler_revision": "z0intelligence@deadbeef",
        "router_revision": "none",
        "cold_or_warm": "warm_invocation",
        "load_ms": 0.0,
        "ttft_ms": 90.0,
        "decision_ms": 200.0,
        "total_ms": 200.0,
        "tokens_in": 500,
        "tokens_out": 20,
        "tok_s": 150.0,
        "peak_vram_mib": 3000,
        "selected_action": "fs.read",
        "distribution": {"fs.read": 1.0, "fs.list": 0.0, "abstain": 0.0},
        "distribution_source": "one_hot",
        "confidence": None,
        "margin": 1.0,
        "entropy": 0.0,
        "abstained": False,
        "invalid_call": False,
        "verification": {"kind": "fixture_gold", "gold_action": "fs.read"},
        "correct": True,
        "dangerous_exposed": False,
        "dangerous_selected": False,
        "failure_retry": {"attempts": 1, "error": None},
        "repetition": 0,
        "source_fixture_revision": "sha256:fixture",
        "deterministic_solution": None,
        "accounting_authority": "tokenomics",
    }
    row.update(overrides)
    return row


class LabelHierarchyTests(unittest.TestCase):
    def test_gold_requires_a_deterministic_verifier(self):
        with self.assertRaises(EpisodeError):
            Episode(
                episode_id="e1",
                state_before={}, available_evidence=[], legal_actions=[],
                candidate_models=[], chosen_action=None, chosen_model=None,
                teacher_distribution=None, observation_before=None, action={},
                observation_after=None, state_after=None, latencies={}, resources={},
                outcome={}, verification={"kind": "model_opinion"},
                source={}, timestamp="", privacy_class="local_only",
                label_source="gold", label_confidence=1.0, verified_success=True,
            )

    def test_verified_success_is_refused_for_non_gold(self):
        for source in ("silver", "teacher"):
            with self.assertRaises(EpisodeError):
                Episode(
                    episode_id="e1", state_before={}, available_evidence=[],
                    legal_actions=[], candidate_models=[], chosen_action=None,
                    chosen_model=None, teacher_distribution=None,
                    observation_before=None, action={}, observation_after=None,
                    state_after=None, latencies={}, resources={}, outcome={},
                    verification={"kind": "fixture_gold"}, source={}, timestamp="",
                    privacy_class="local_only", label_source=source,
                    label_confidence=0.5, verified_success=True,
                )

    def test_fixture_gold_receipt_becomes_a_gold_episode(self):
        episode = compile_episode(_receipt())
        self.assertEqual(episode.label_source, "gold")
        self.assertTrue(episode.verified_success)
        self.assertEqual(episode.privacy_class, "local_only")
        self.assertIsNone(episode.resources["cost_usd"])
        self.assertEqual(episode.resources["accounting_authority"], "tokenomics")

    def test_unverified_receipt_never_claims_success(self):
        episode = compile_episode(_receipt(verification={"kind": "none"}, correct=True))
        self.assertEqual(episode.label_source, "silver")
        self.assertIsNone(episode.verified_success)

    def test_scorer_distribution_becomes_a_teacher_label(self):
        episode = compile_episode(_receipt(
            distribution={"fs.read": 0.6, "fs.list": 0.4, "abstain": 0.0},
            distribution_source="scorer", confidence=0.6,
            verification={"kind": "none"}, correct=None,
        ))
        self.assertEqual(episode.label_source, "teacher")
        self.assertEqual(episode.teacher_distribution["fs.read"], 0.6)
        self.assertIsNone(episode.verified_success)


class SelectiveTeacherTests(unittest.TestCase):
    def _episodes(self):
        good = compile_episode(_receipt(trace_id="a"))
        # a state no arm resolves
        bad_rows = [
            _receipt(trace_id=f"b{i}", state_id="slm_uncertain", arm=f"arm{i}",
                     correct=False, selected_action="fs.read")
            for i in range(3)
        ]
        return [good] + [compile_episode(r) for r in bad_rows]

    def test_selection_is_not_the_whole_corpus(self):
        episodes = self._episodes()
        selection = selective_teacher_candidates(episodes)
        self.assertLess(selection["n_selected"], len(episodes))
        reasons = {r for row in selection["selected"] for r in row["reasons"]}
        self.assertIn("ambiguity", reasons)

    def test_deterministic_arm_is_never_nominated_for_uncertainty(self):
        # A no-model control returning an all-zero distribution is the absence of
        # a model, not model uncertainty, so it must never be nominated for
        # teacher inference on that basis.
        episodes = [compile_episode(_receipt(arm="deterministic.compiler_only",
                                             arm_alias=None, model_id=None,
                                             distribution={"fs.read": 0.0, "fs.list": 0.0},
                                             distribution_source="none",
                                             selected_action=None, abstained=True,
                                             legal_actions=["fs.read", "fs.list"],
                                             correct=False))]
        selection = selective_teacher_candidates(episodes)
        for row in selection["selected"]:
            self.assertNotIn("high_information", row["reasons"])

    def test_resolved_state_is_not_nominated(self):
        episodes = [compile_episode(_receipt(trace_id="z"))]
        selection = selective_teacher_candidates(episodes)
        self.assertEqual(selection["n_selected"], 0)


class SplitTests(unittest.TestCase):
    def _episodes(self, n_tasks: int = 20):
        rows = []
        for i in range(n_tasks):
            for rep in range(2):
                rows.append(_receipt(
                    state_id=f"task_{i:02d}",
                    trace_id=f"t{i}-{rep}",
                    observed_at=f"2026-09-{(i % 27) + 1:02d}T10:00:00Z",
                    state_family=["tool_selection", "recovery", "security",
                                  "schema", "routing"][i % 5],
                ))
        return [compile_episode(r) for r in rows]

    def test_chronological_split_keeps_tasks_whole(self):
        plan = build_split_plan(self._episodes(), sealed_size=4)
        seen: dict[str, set[str]] = {}
        for row in plan["assignments"]:
            seen.setdefault(row["task_id"], set()).add(row["chronological_bucket"])
        for task, buckets in seen.items():
            self.assertEqual(len(buckets), 1, f"{task} leaked across buckets: {buckets}")

    def test_buckets_are_disjoint_and_complete(self):
        episodes = self._episodes()
        plan = build_split_plan(episodes, sealed_size=4)
        ids = [row["episode_id"] for row in plan["assignments"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(episodes))
        self.assertLessEqual(set(plan["bucket_counts"]),
                             {"train", "validation", "confirm", "ood",
                              "sealed_human_audited"})

    def test_ood_holds_out_whole_families(self):
        plan = build_split_plan(self._episodes(), sealed_size=4)
        held = set(plan["ood"]["holdout_families"])
        self.assertTrue(held)
        for row in plan["assignments"]:
            if row["task_family"] in held and row["bucket"] not in (
                "sealed_human_audited", "ood"
            ):
                self.fail(f"{row['task_id']} in a held-out family stayed in {row['bucket']}")

    def test_sealed_set_is_never_in_train(self):
        plan = build_split_plan(self._episodes(), sealed_size=4)
        for row in plan["assignments"]:
            if row["bucket"] == "sealed_human_audited":
                self.assertEqual(row["chronological_bucket"], "confirm")

    def test_split_digest_is_stable(self):
        episodes = self._episodes()
        first = build_split_plan(episodes, sealed_size=4)["split_digest"]
        second = build_split_plan(episodes, sealed_size=4)["split_digest"]
        self.assertEqual(first, second)


class CompileEpisodesTests(unittest.TestCase):
    def test_round_trip_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for i in range(3):
                    fh.write(json.dumps(_receipt(trace_id=f"t{i}")) + "\n")
            episodes = compile_episodes(path)
            self.assertEqual(len(episodes), 3)
            self.assertEqual(label_census(episodes), {"gold": 3})

    def test_missing_state_text_is_joined_from_the_fixture_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = Path(tmp) / "observations.jsonl"
            obs.write_text(json.dumps(_receipt()) + "\n", encoding="utf-8")
            fixtures = Path(tmp) / "examples.jsonl"
            fixtures.write_text(
                json.dumps({"fixture_id": "one_obvious_tool",
                            "state": "The user asked to see /etc/hostname.",
                            "family": "tool_selection"}) + "\n",
                encoding="utf-8",
            )
            episodes = compile_episodes(obs, fixtures_path=fixtures)
            self.assertEqual(episodes[0].state_before["text"],
                             "The user asked to see /etc/hostname.")
            self.assertEqual(episodes[0].task_family, "tool_selection")


if __name__ == "__main__":
    unittest.main()
