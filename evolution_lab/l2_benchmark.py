"""First L2 specialist battery: ``recovery_action``.

Why this card (locked pick, 2026-09-18):
- ``label_quality=high`` and external gym already exist (HermesRecoveryEnv).
- Locked P0 splits under ``data/p0`` (seed 20260912) are evolver-blind.
- ``needs_verification`` sealed L0 is too thin (n≈14) and needs test-join instrumentation.
- ``delegate_gating`` stays L0 fingerprint only until wall-clock join exists.

Evidence levels (capability_schema):
- L0 imitation on labels
- L1 teacher
- **L2 outcome** — closed-loop reward / subsequent success in gym (this module)
- L3 harness canary with fallback measurement

Promote: shadow → canary only when confirm/ood/closed_loop gates pass.
OMP/z0int-bridge stay ``log_only`` until the canary marker exists under ``~/.z0int/specialists/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .coverage_metric import binary_gate_metrics
from .dagger import mean_closed_loop_reward
from .gym import make_env
from .schema import ACTIONS
from .slice_informativeness import (
    GATE_UNINFORMATIVE,
    assess_slice,
    gate_status,
    summarise as summarise_slices,
)
from .splits import load_splits
from .student_bundle import bundle_exists, bundle_meta, load_bundle

CAPABILITY_ID = "recovery_action"
SCHEMA = "z0int.l2_benchmark.v1"
# Locked gates from gpu_evolve CPU judge / P0 contract
GATE_CONFIRM = 0.95
GATE_VAL = 0.95
GATE_OOD = 0.85
GATE_CLOSED_LOOP = 1.0 - 1e-9

#: These gates read the per-EPISODE unit. `results/recovery-compare.json` reports
#: the same frozen bundle per STEP (0.8385) while this file recorded 1.000 per
#: episode and promoted on it. Both numbers are real; the differing unit was
#: never declared, which is how they came to coexist. Declared here, and carried
#: into the report, so a comparison contract can reject the pair.
METRIC_UNIT = "accuracy"
AGGREGATION_UNIT = "per_episode"


def _home() -> Path:
    import os

    override = os.environ.get("Z0INT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".z0int").resolve()


def _success(pred: np.ndarray, split) -> float:
    y = np.stack([ep.labels[-1] for ep in split])
    if len(y) == 0:
        return 0.0
    return float((np.asarray(pred) == y).mean())


def _majority_predict(train_split, eval_split) -> np.ndarray:
    ytr = np.stack([ep.labels[-1] for ep in train_split])
    # mode
    vals, counts = np.unique(ytr, return_counts=True)
    maj = int(vals[int(np.argmax(counts))])
    return np.full(len(eval_split), maj, dtype=np.int64)


def _rule_predict(eval_split) -> np.ndarray:
    """Teacher rule on last frame of each episode."""
    from .task import teacher_action

    out = []
    for ep in eval_split:
        try:
            out.append(int(teacher_action(np.asarray(ep.frames[-1]))))
        except Exception:
            out.append(int(ep.labels[-1]))
    return np.asarray(out, dtype=np.int64)


def _load_student():
    from .advise import _local_plasticity_student

    return _local_plasticity_student()



def _split_metrics(name: str, pred: np.ndarray, split) -> dict[str, Any]:
    y = np.stack([ep.labels[-1] for ep in split]).astype(np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    acc = float((pred == y).mean()) if len(y) else 0.0
    # multiclass → one-vs-rest micro summary via accuracy + per-action support
    by_action: dict[str, dict[str, Any]] = {}
    for i, act in enumerate(ACTIONS):
        m = y == i
        if not m.any():
            continue
        by_action[act] = {
            "n": int(m.sum()),
            "recall": float((pred[m] == y[m]).mean()),
        }
    # binary "is page_human" as high-cost gate example
    page_i = list(ACTIONS).index("page_human") if "page_human" in ACTIONS else None
    gate = None
    if page_i is not None:
        gate = binary_gate_metrics(y, pred, positive=page_i)
    return {
        "split": name,
        "n": int(len(y)),
        "accuracy": acc,
        "by_action_recall": by_action,
        "page_human_gate": gate,
    }


def run_l2_recovery_benchmark(
    *,
    closed_loop_seeds: int = 24,
    write: bool = True,
    promote: bool = True,
) -> dict[str, Any]:
    """Sealed L2 battery for recovery_action on locked P0 + closed-loop gym."""
    t0 = time.time()
    data = load_splits()
    root = Path(__file__).resolve().parents[1]
    has_bundle = bundle_exists(root)
    meta = bundle_meta(root) if has_bundle else None

    controls: dict[str, dict[str, Any]] = {}
    # majority
    maj_c = _majority_predict(data.train, data.confirm)
    maj_v = _majority_predict(data.train, data.val)
    maj_o = _majority_predict(data.train, data.ood)
    controls["majority"] = {
        "confirm": _split_metrics("confirm", maj_c, data.confirm),
        # `val` is gated at GATE_VAL like the others, so it needs a baseline to be
        # gated *against*. Without one the threshold check had nothing to
        # discriminate from and passed vacuously.
        "val": _split_metrics("val", maj_v, data.val),
        "ood": _split_metrics("ood", maj_o, data.ood),
        "closed_loop_reward": None,
    }
    # rule / teacher
    rule_c = _rule_predict(data.confirm)
    rule_o = _rule_predict(data.ood)
    controls["rule_teacher"] = {
        "confirm": _split_metrics("confirm", rule_c, data.confirm),
        "ood": _split_metrics("ood", rule_o, data.ood),
        "closed_loop_reward": None,
    }
    student_block: dict[str, Any] | None = None
    closed_loop = None
    if has_bundle:
        genome, student = _load_student()
        pred_c = student.predict_fn(data.confirm)
        pred_o = student.predict_fn(data.ood)
        pred_v = student.predict_fn(data.val)
        hist = int(getattr(genome.architecture, "history", None) or (meta or {}).get("history") or 8)
        env = make_env("hermes_recovery", delayed_cue=True, history=hist)
        closed_loop = float(mean_closed_loop_reward(env, student, list(range(closed_loop_seeds))))
        student_block = {
            "confirm": _split_metrics("confirm", pred_c, data.confirm),
            "val": _split_metrics("val", pred_v, data.val),
            "ood": _split_metrics("ood", pred_o, data.ood),
            "closed_loop_reward": closed_loop,
            "n_params": int(getattr(student, "n_params", 0) or (meta or {}).get("n_params") or 0),
            "bundle_meta": meta,
            "genome_id": getattr(genome, "id", None),
        }
        controls["student"] = student_block

    confirm_acc = float((student_block or {}).get("confirm", {}).get("accuracy") or 0.0)
    val_acc = float((student_block or {}).get("val", {}).get("accuracy") or 0.0)
    ood_acc = float((student_block or {}).get("ood", {}).get("accuracy") or 0.0)
    cl = float(closed_loop) if closed_loop is not None else 0.0

    # --- slice informativeness -------------------------------------------
    # A gate that compares a candidate to a threshold only means something if
    # the slice can discriminate. On 2026-09-21 this capability was promoted to
    # canary with promote_allowed=true on an OOD slice whose 32/32 labels are
    # `escalate`, so the majority baseline scored 1.000 and cleared `ood_ge`.
    def _gold(split) -> list[int]:
        return [int(ep.labels[-1]) for ep in split]

    train_gold = _gold(data.train)
    slice_assessments = []
    for slice_id, split, threshold in (
        ("confirm", data.confirm, GATE_CONFIRM),
        ("val", data.val, GATE_VAL),
        ("ood", data.ood, GATE_OOD),
    ):
        maj = ((controls.get("majority") or {}).get(slice_id) or {}).get("accuracy")
        slice_assessments.append(
            assess_slice(
                slice_id=slice_id,
                gold_labels=_gold(split),
                fit_labels=train_gold,
                majority_accuracy=(float(maj) if maj is not None else None),
                metric_unit=METRIC_UNIT,
                aggregation_unit=AGGREGATION_UNIT,
                gate_threshold=threshold,
                declared_class_count=len(ACTIONS),
            )
        )

    gates = {
        "confirm_ge": GATE_CONFIRM,
        "val_ge": GATE_VAL,
        "ood_ge": GATE_OOD,
        "closed_loop_ge": GATE_CLOSED_LOOP,
        "confirm_ok": confirm_acc + 1e-12 >= GATE_CONFIRM,
        "val_ok": val_acc + 1e-12 >= GATE_VAL,
        "ood_ok": ood_acc + 1e-12 >= GATE_OOD,
        "closed_loop_ok": cl + 1e-12 >= GATE_CLOSED_LOOP,
        "bundle_present": has_bundle,
    }
    thresholds_passed = bool(
        gates["bundle_present"]
        and gates["confirm_ok"]
        and gates["val_ok"]
        and gates["ood_ok"]
        and gates["closed_loop_ok"]
    )
    slices_informative = all(s.informative for s in slice_assessments)
    gates["thresholds_passed"] = thresholds_passed
    gates["slices_informative"] = slices_informative
    gates["informative_slices"] = [s.slice_id for s in slice_assessments if s.informative]
    gates["uninformative_slices"] = [s.slice_id for s in slice_assessments if not s.informative]
    # Pass now requires both: the thresholds cleared AND the slices able to tell
    # the difference. An uninformative slice outranks a fail — it is an absent
    # test, not a failed one.
    gates["pass"] = bool(thresholds_passed and slices_informative)
    gates["gate_status"] = gate_status(slice_assessments, thresholds_passed=thresholds_passed)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "capability_id": CAPABILITY_ID,
        "evidence_level": "L2_outcome",
        "ts": time.time(),
        "elapsed_s": time.time() - t0,
        "pick_rationale": (
            "recovery_action: high label_quality, external gym, locked P0 splits; "
            "needs_verification deferred (thin sealed L0 + test-join gap)"
        ),
        "splits": {
            "source": "data/p0 locked seed 20260912",
            "n_train": len(data.train),
            "n_val": len(data.val),
            "n_confirm": len(data.confirm),
            "n_ood": len(data.ood),
        },
        "actions": list(ACTIONS),
        "metric_unit": METRIC_UNIT,
        "aggregation_unit": AGGREGATION_UNIT,
        "unit_note": (
            "These gates read the per-episode unit. results/recovery-compare.json "
            "reports the same bundle per step (0.8385). Declared so the two can no "
            "longer be compared without a mismatch being visible."
        ),
        "controls": controls,
        "slice_informativeness": summarise_slices(slice_assessments),
        "gates": gates,
        "execution_policy": {
            "current": "canary" if gates["pass"] else "shadow",
            "omp_hooks": "log_only_until_canary_marker",
            "promote_allowed": bool(promote and gates["pass"]),
            "gate_status": gates["gate_status"],
            "blocked_reason": (
                None
                if gates["pass"]
                else (
                    "uninformative promotion slice(s): "
                    + ", ".join(gates["uninformative_slices"])
                    if gates["uninformative_slices"]
                    else "gate thresholds not met"
                )
            ),
        },
    }

    # dual-write receipt summary
    try:
        from z0int.receipt import append_receipt, build_receipt  # type: ignore

        append_receipt(
            build_receipt(
                capability_id=CAPABILITY_ID,
                provider="local_mb",
                model="recovery_student",
                prediction="l2_benchmark",
                confidence=confirm_acc,
                action_taken="canary" if gates["pass"] else "shadow",
                route="local",
                execution="canary" if gates["pass"] else "log_only",
                extra={"l2_report_schema": SCHEMA, "gates_pass": gates["pass"], "closed_loop": cl},
            )
        )
    except Exception:
        pass

    out_dir = _home() / "benchmarks"
    spec_dir = _home() / "specialists"
    # The 2026-09-21 artifact is historical evidence for a gate that promoted on
    # an uninformative slice. It is NOT overwritten: its recorded result stands,
    # and the revised report is written beside it. A re-run must not be able to
    # rewrite the record of what previously passed.
    legacy_path = out_dir / "recovery_action_l2.json"
    report_path = out_dir / "recovery_action_l2.revised.json"
    canary_path = spec_dir / "recovery_action.canary.json"
    report["supersedes"] = str(legacy_path) if legacy_path.is_file() else None
    if legacy_path.is_file():
        try:
            prev = json.loads(legacy_path.read_text(encoding="utf-8"))
            report["superseded_verdict"] = {
                "gates_pass": (prev.get("gates") or {}).get("pass"),
                "promote_allowed": (prev.get("execution_policy") or {}).get("promote_allowed"),
                "note": (
                    "The superseded run passed on an OOD slice whose labels are a single "
                    "class, so its majority baseline scored 1.000 and cleared ood_ge=0.85."
                ),
            }
        except (OSError, json.JSONDecodeError):
            report["superseded_verdict"] = {"note": "legacy artifact present but unreadable"}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_dir.mkdir(parents=True, exist_ok=True)
        # Resolve the canary outcome BEFORE writing, so every field below reaches
        # disk. Previously the report was written first and the stale-marker flag
        # was set afterwards, so the flag existed only in the returned dict and
        # never in the artifact a later reader would find.
        if promote and gates["pass"]:
            marker = {
                "schema": "z0int.specialist_canary.v1",
                "capability_id": CAPABILITY_ID,
                "ts": time.time(),
                "gates": gates,
                "from_report": str(report_path),
                "execution": "canary",
                "note": "OMP hooks may canary this capability; default remains log_only for others",
            }
            canary_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
            report["canary_marker"] = str(canary_path)
        elif canary_path.exists() and not gates["pass"]:
            # do not delete existing canary automatically — fail closed on new run only
            report["canary_marker_stale"] = str(canary_path)
            report["canary_marker_stale_reason"] = (
                f"an existing canary marker remains from a run that passed under gate "
                f"semantics this run does not ({gates['gate_status']}); it is deliberately "
                "not deleted, but it must not be read as a current endorsement"
            )
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    return report


def canary_active(capability_id: str = CAPABILITY_ID) -> bool:
    p = _home() / "specialists" / f"{capability_id}.canary.json"
    if not p.is_file():
        return False
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(raw.get("execution") == "canary")
