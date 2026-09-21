"""Tests for the capacity-fill queue hand-off (issue kvnloo/evolution-lab#23).

The queue is the small interface a capacity owner (Kerdoios) asks: "what useful
compatible work is queued, and what would consume it?"  These tests pin the two
contracts that matter: priority follows the agreed useful-work order, and no
work is fabricated -- including never offering protected confirm/OOD/held-out
work as background fill.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evolution_lab import cli
from evolution_lab.capacity_queue import (
    FILL_SPLITS,
    PROTECTED_SPLITS,
    WORK_CLASSES,
    EstimatedUse,
    QueueError,
    WorkConstraints,
    WorkItem,
    default_queue_path,
    is_fill_split,
    is_protected_split,
    load_queue,
    plan_capacity,
    priority_of,
)

#: The useful-work order from the ownership brief. Strings, not indexes, so a
#: reordering is a test failure rather than a silent priority change.
EXPECTED_WORK_ORDER = (
    "dsh_hermes",
    "qroute_counterfactual",
    "router_eval",
    "distillation_example",
    "agent0_curriculum",
    "jev_tiny_router_data",
    "queued_research",
)


def make_item(item_id: str = "item-1", **kwargs) -> WorkItem:
    base = {
        "work_class": "router_eval",
        "summary": "frozen eval",
        "provenance": "data/tool_tournament/v1/fixtures.jsonl",
    }
    base.update(kwargs)
    return WorkItem(item_id=item_id, **base)  # type: ignore[arg-type]


class PriorityOrderTests(unittest.TestCase):
    def test_work_class_order_matches_brief(self):
        self.assertEqual(WORK_CLASSES, EXPECTED_WORK_ORDER)
        self.assertEqual([priority_of(c) for c in WORK_CLASSES], list(range(len(WORK_CLASSES))))

    def test_unknown_work_class_is_rejected(self):
        with self.assertRaises(QueueError):
            priority_of("not_a_work_class")

    def test_priority_is_lower_for_more_useful_work(self):
        self.assertLess(priority_of("dsh_hermes"), priority_of("queued_research"))


class AntiFillerTests(unittest.TestCase):
    def test_item_without_provenance_is_rejected(self):
        with self.assertRaises(QueueError):
            make_item(provenance="")

    def test_item_without_summary_is_rejected(self):
        with self.assertRaises(QueueError):
            make_item(summary="")

    def test_unknown_work_class_is_rejected(self):
        with self.assertRaises(QueueError):
            make_item(work_class="filler")

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(QueueError):
            make_item(constraints=WorkConstraints(split="whatever"))

    def test_negative_estimates_are_rejected(self):
        with self.assertRaises(QueueError):
            make_item(estimate=EstimatedUse(calls=-1))

    def test_empty_queue_produces_no_work(self):
        plan = plan_capacity([], provider="groq")
        self.assertEqual(plan["selected"], [])
        self.assertEqual(plan["totals"]["calls"], 0)
        self.assertIn("empty", plan["empty_reason"])
        self.assertIn("no work is generated", plan["filler_policy"])


class SplitGuardTests(unittest.TestCase):
    def test_protected_splits_are_known(self):
        for split in ("confirm", "ood", "sealed", "sealed_human_audited", "holdout", "future"):
            self.assertTrue(is_protected_split(split), split)
            self.assertFalse(is_fill_split(split), split)
        self.assertNotIn("confirm", FILL_SPLITS)
        self.assertNotIn("ood", FILL_SPLITS)

    def test_protected_item_is_skipped_by_default(self):
        item = make_item(item_id="confirm-eval", constraints=WorkConstraints(split="confirm"))
        plan = plan_capacity([item], provider="groq")
        self.assertEqual([s["item_id"] for s in plan["selected"]], [])
        self.assertEqual(plan["skipped"][0]["reason"], "protected_split_certification_only")

    def test_protected_item_runs_only_when_asked(self):
        item = make_item(item_id="ood-eval", constraints=WorkConstraints(split="ood"))
        plan = plan_capacity([item], provider="groq", include_protected=True)
        self.assertEqual([s["item_id"] for s in plan["selected"]], ["ood-eval"])

    def test_fill_safe_flag_follows_split(self):
        self.assertTrue(make_item(constraints=WorkConstraints(split="train")).fill_safe)
        self.assertFalse(make_item(constraints=WorkConstraints(split="holdout")).fill_safe)


class CompatibilityTests(unittest.TestCase):
    def test_provider_mismatch_is_skipped(self):
        item = make_item(constraints=WorkConstraints(providers=("local",)))
        plan = plan_capacity([item], provider="groq")
        self.assertEqual(plan["selected"], [])
        self.assertEqual(plan["skipped"][0]["reason"], "incompatible_provider_or_model")

    def test_provider_and_model_constraints_match(self):
        item = make_item(
            constraints=WorkConstraints(providers=("groq",), models=("qwen3.5-4b",)),
        )
        self.assertEqual(
            [s["item_id"] for s in plan_capacity([item], provider="groq", model="qwen3.5-4b")["selected"]],
            ["item-1"],
        )
        self.assertEqual(plan_capacity([item], provider="groq", model="hammer2.1-3b")["selected"], [])

    def test_plan_orders_by_priority_then_id(self):
        items = [
            make_item(item_id="b", work_class="queued_research"),
            make_item(item_id="a", work_class="router_eval"),
            make_item(item_id="c", work_class="dsh_hermes"),
        ]
        plan = plan_capacity(items, provider="groq")
        self.assertEqual([s["item_id"] for s in plan["selected"]], ["c", "a", "b"])


class BudgetTests(unittest.TestCase):
    def test_totals_sum_selected_estimates(self):
        items = [
            make_item(
                item_id="a",
                estimate=EstimatedUse(calls=2, prompt_tokens=100, completion_tokens=50, cost_usd=0.25),
            ),
            make_item(
                item_id="b",
                estimate=EstimatedUse(calls=3, prompt_tokens=200, completion_tokens=100, cost_usd=0.5),
            ),
        ]
        plan = plan_capacity(items, provider="groq")
        self.assertEqual(plan["totals"]["calls"], 5)
        self.assertEqual(plan["totals"]["tokens"], 450)
        self.assertAlmostEqual(plan["totals"]["cost_usd"], 0.75)

    def test_token_budget_is_respected(self):
        items = [
            make_item(item_id="big", work_class="dsh_hermes", estimate=EstimatedUse(prompt_tokens=1000)),
            make_item(item_id="small", work_class="router_eval", estimate=EstimatedUse(prompt_tokens=10)),
        ]
        plan = plan_capacity(items, provider="groq", max_tokens=50)
        self.assertEqual([s["item_id"] for s in plan["selected"]], ["small"])
        self.assertEqual(plan["skipped"][0]["reason"], "budget_exhausted_tokens")

    def test_cost_budget_is_respected(self):
        item = make_item(estimate=EstimatedUse(cost_usd=1.0))
        plan = plan_capacity([item], provider="groq", max_cost_usd=0.1)
        self.assertEqual(plan["selected"], [])
        self.assertEqual(plan["skipped"][0]["reason"], "budget_exhausted_cost")


class RoundTripTests(unittest.TestCase):
    def test_item_round_trips_through_dict(self):
        item = make_item(
            item_id="rt",
            constraints=WorkConstraints(providers=("groq",), task_families=("coding",), split="validation"),
            estimate=EstimatedUse(calls=1, prompt_tokens=2, completion_tokens=3, cost_usd=0.0, gpu_ms=4),
            created_by="tests",
        )
        restored = WorkItem.from_dict(item.to_dict())
        self.assertEqual(restored.to_dict(), item.to_dict())

    def test_load_missing_file_is_empty(self):
        missing = Path(tempfile.mkdtemp()) / "nope.json"
        self.assertEqual(load_queue(missing), [])


class DefaultQueueTests(unittest.TestCase):
    """The shipped queue must be explicit, non-filler and provenance-backed."""

    @classmethod
    def setUpClass(cls):
        cls.path = default_queue_path()
        cls.root = cls.path.parents[2]
        cls.items = load_queue(cls.path)

    def test_default_queue_exists(self):
        self.assertTrue(self.path.is_file(), self.path)

    def test_default_queue_is_non_empty_and_unique(self):
        self.assertGreater(len(self.items), 0)
        ids = [item.item_id for item in self.items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_queued_work_class_is_covered(self):
        self.assertEqual({item.work_class for item in self.items}, set(WORK_CLASSES))

    def test_default_queue_has_no_protected_fill(self):
        # The shipped default is background fill, so no protected split belongs in it.
        for item in self.items:
            self.assertTrue(item.fill_safe, f"{item.item_id} uses protected split {item.constraints.split}")

    def test_every_provenance_file_exists(self):
        for item in self.items:
            self.assertTrue((self.root / item.provenance).exists(), item.provenance)

    def test_default_plan_selects_by_priority(self):
        plan = plan_capacity(self.items, provider="groq", max_tokens=60000)
        priorities = [s["priority"] for s in plan["selected"]]
        self.assertEqual(priorities, sorted(priorities))
        self.assertTrue(plan["selected"])


class CliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(argv)
        self.assertEqual(code, 0)
        return json.loads(buf.getvalue())

    def test_list_prints_the_queue(self):
        payload = self._run(["capacity-queue", "list"])
        self.assertEqual(payload["schema"], "flyforge.capacity_queue.v1")
        self.assertEqual(payload["count"], len(payload["items"]))

    def test_plan_prints_a_plan(self):
        payload = self._run(["capacity-queue", "plan", "--provider", "groq", "--max-tokens", "60000"])
        self.assertEqual(payload["schema"], "flyforge.capacity_plan.v1")
        self.assertTrue(payload["selected"])
        self.assertLessEqual(payload["totals"]["tokens"], 60000)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
