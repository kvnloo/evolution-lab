"""Capability Miner — what decisions are worth learning?

Mines ~/.z0int episodes (+ optional stream) into CapabilityCards, sealed
BenchmarkSpecs, L0 niche results, and research deliverables under
~/.z0int/research/. Does NOT train new champions; discovery only.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .capability_schema import (
    BenchmarkSpec,
    CapabilityCard,
    CapabilityResult,
    DEFAULT_CONTROLS,
)
from .coverage_metric import coverage_at_precision
from .models import _kc_codes
from .next_action_gpu import EPISODES, FAMILIES, FAM_I, _hash, ridge_acc
from .sealed_split import load_episode_rows, make_session_split, mask_for_bucket

RESEARCH = Path.home() / ".z0int" / "research"
SEED_CARDS: list[dict[str, Any]] = [
    {
        "id": "delegate_gating",
        "description": "Should this turn be delegated to a subagent/scout?",
        "stage": "routing",
        "trigger": "agent considering next action under multi-step work",
        "input_contract": ["prompt", "prev_families", "session_depth"],
        "output_contract": ["yes", "no"],
        "verifier": "historical next family==DELEGATE (L0); later wall-clock/task success (L2/L3)",
        "label_sources": ["historical", "outcome"],
        "failure_cost": "medium",
        "temporal_dependency": "low",
        "label_quality": "medium",
    },
    {
        "id": "next_action_family",
        "description": "Which coarse tool family comes next? (current L0 task)",
        "stage": "routing",
        "trigger": "every agent turn",
        "input_contract": ["prompt", "prev_families"],
        "output_contract": list(FAMILIES),
        "verifier": "historical next family (L0 only; not optimal-action proof)",
        "label_sources": ["historical"],
        "failure_cost": "low",
        "temporal_dependency": "medium",
        "label_quality": "low",
    },
    {
        "id": "needs_verification",
        "description": "After an edit, is a test/execute check needed next?",
        "stage": "verification",
        "trigger": "previous action family was EDIT",
        "input_contract": ["prompt", "prev_families", "last_tool"],
        "output_contract": ["yes", "no"],
        "verifier": "next family in {EXECUTE,VERIFY} (L0); later test pass/fail (L2)",
        "label_sources": ["historical", "outcome"],
        "failure_cost": "high",
        "temporal_dependency": "low",
        "label_quality": "medium",
    },
    {
        "id": "retry_execute",
        "description": "After EXECUTE, should we execute again (retry/loop)?",
        "stage": "recovery",
        "trigger": "previous action family was EXECUTE",
        "input_contract": ["prompt", "prev_families", "tool_result"],
        "output_contract": ["yes", "no"],
        "verifier": "next family==EXECUTE (L0); exit code improvement (L2)",
        "label_sources": ["historical", "outcome"],
        "failure_cost": "medium",
        "temporal_dependency": "medium",
        "label_quality": "low",
    },
    {
        "id": "continue_same_family",
        "description": "Stay in the same action family as the previous step?",
        "stage": "planning",
        "trigger": "non-empty prev families",
        "input_contract": ["prompt", "prev_families"],
        "output_contract": ["yes", "no"],
        "verifier": "next family == prev[-1]",
        "label_sources": ["historical"],
        "failure_cost": "low",
        "temporal_dependency": "low",
        "label_quality": "medium",
    },
    {
        "id": "read_vs_edit",
        "description": "When leaving search/read, is edit appropriate vs more search/execute?",
        "stage": "editing",
        "trigger": "prev family was READ_SEARCH",
        "input_contract": ["prompt", "prev_families"],
        "output_contract": ["EDIT", "READ_SEARCH", "EXECUTE", "other"],
        "verifier": "historical next family among {EDIT,READ_SEARCH,EXECUTE}",
        "label_sources": ["historical"],
        "failure_cost": "medium",
        "temporal_dependency": "medium",
        "label_quality": "low",
    },
    {
        "id": "context_file_relevance",
        "description": "Which files/paths deserve attention (retrieval specialist slot)?",
        "stage": "context",
        "trigger": "search/read phase",
        "input_contract": ["prompt", "candidate_paths", "repo"],
        "output_contract": ["paths_or_ranges"],
        "verifier": "paths ultimately edited or read in successful trajectory (L2)",
        "label_sources": ["outcome", "git"],
        "failure_cost": "medium",
        "temporal_dependency": "low",
        "label_quality": "high",
        "instrumentation_gap": True,
    },
    {
        "id": "task_complete",
        "description": "Is the task done (stop vs continue)?",
        "stage": "completion",
        "trigger": "after execute/verify/respond consideration",
        "input_contract": ["prompt", "recent_outcomes", "todo_state"],
        "output_contract": ["yes", "no", "uncertain"],
        "verifier": "no further tool use before user turn; user acceptance (L2/L3)",
        "label_sources": ["historical", "user"],
        "failure_cost": "high",
        "temporal_dependency": "medium",
        "label_quality": "low",
        "instrumentation_gap": True,
    },
    {
        "id": "recovery_action",
        "description": "Retry / restart_sandbox / escalate / noop / page_human (P0 recovery).",
        "stage": "recovery",
        "trigger": "tool/process failure event",
        "input_contract": ["sanitized_event_fields"],
        "output_contract": ["retry", "restart_sandbox", "escalate", "noop", "page_human"],
        "verifier": "external recovery gym / subsequent success (L2/L3)",
        "label_sources": ["outcome", "gym"],
        "failure_cost": "high",
        "temporal_dependency": "low",
        "label_quality": "high",
    },
    {
        "id": "skill_routing",
        "description": "Which skill applies for this prompt?",
        "stage": "routing",
        "trigger": "before_agent_start / skill suggestion",
        "input_contract": ["prompt", "installed_skills"],
        "output_contract": ["skill_id_or_none"],
        "verifier": "skill activation + task outcome (L2); Jev soft labels (L1)",
        "label_sources": ["jev", "historical"],
        "failure_cost": "low",
        "temporal_dependency": "low",
        "label_quality": "medium",
    },
]


def _days_span(rows: list[dict[str, Any]]) -> float:
    ts = [float(r["ts"]) for r in rows if r.get("ts") is not None]
    if len(ts) < 2:
        return 1.0
    return max((max(ts) - min(ts)) / 86400.0, 1.0)


def _load_pack(path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    g0 = root / "runs" / "gpu-evolve" / "gen0_champion_backup.npz"
    cand = path or (g0 if g0.exists() else root / "data" / "next_action" / "champion.npz")
    data = np.load(cand)
    # prefer 64-d gen-0 for classic L0 fingerprints
    if int(data["W_pn_kc"].shape[0]) != 64 and g0.exists():
        data = np.load(g0)
    return {
        "W_pn_kc": data["W_pn_kc"],
        "W_kc_mbon": data["W_kc_mbon"],
        "k_winners": int(data["k_winners"]),
        "pn_dim": int(data["W_pn_kc"].shape[0]),
        "acc": float(data["acc"]) if "acc" in data.files else None,
        "path": str(cand if int(np.load(cand)["W_pn_kc"].shape[0]) == 64 or not g0.exists() else g0),
    }


def _build_xy(rows: list[dict[str, Any]], dim: int = 64) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    sessions: list[str] = []
    for r in rows:
        fam = r.get("family")
        if fam not in FAM_I:
            continue
        text = r.get("text") or r.get("user") or ""
        prev = r.get("prev") or []
        if prev:
            text = text + " " + " ".join(str(x) for x in prev)
        xs.append(_hash(text, dim))
        ys.append(FAM_I[fam])
        sessions.append(str(r.get("session") or ""))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int32), sessions


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    return e / e.sum(axis=1, keepdims=True)


def mine_corpus_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fam = Counter(r.get("family") for r in rows)
    tier = Counter(r.get("tier") or "" for r in rows)
    tool = Counter(str(r.get("tool") or "")[:48] for r in rows)
    sess = Counter(str(r.get("session") or "") for r in rows)
    trans: Counter[tuple] = Counter()
    for r in rows:
        prev = r.get("prev") or []
        last = prev[-1] if prev else None
        trans[(last, r.get("family"))] += 1
    days = _days_span(rows)
    return {
        "n_events": len(rows),
        "n_sessions": len(sess),
        "median_session_len": int(sorted(sess.values())[len(sess) // 2]) if sess else 0,
        "span_days": days,
        "events_per_day": len(rows) / days,
        "sessions_per_day": len(sess) / days,
        "family": dict(fam.most_common()),
        "tier": dict(tier),
        "top_tools": tool.most_common(20),
        "top_transitions": [(list(k), v) for k, v in trans.most_common(20)],
        "gold_is_tool_success_not_optimal": True,
    }


def _proxy_labels(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Return per-row labels; -1 = not in trigger set."""
    n = len(rows)
    out: dict[str, np.ndarray] = {
        "delegate_gating": np.full(n, -1, dtype=np.int32),
        "needs_verification": np.full(n, -1, dtype=np.int32),
        "retry_execute": np.full(n, -1, dtype=np.int32),
        "continue_same_family": np.full(n, -1, dtype=np.int32),
        "next_action_family": np.full(n, -1, dtype=np.int32),
    }
    for i, r in enumerate(rows):
        fam = r.get("family")
        if fam not in FAM_I:
            continue
        out["next_action_family"][i] = FAM_I[fam]
        out["delegate_gating"][i] = 1 if fam == "DELEGATE" else 0
        prev = r.get("prev") or []
        if prev:
            out["continue_same_family"][i] = 1 if fam == prev[-1] else 0
            if prev[-1] == "EDIT":
                out["needs_verification"][i] = 1 if fam in ("EXECUTE", "VERIFY") else 0
            if prev[-1] == "EXECUTE":
                out["retry_execute"][i] = 1 if fam == "EXECUTE" else 0
    return out


def evaluate_l0_niches(
    rows: list[dict[str, Any]],
    split,
    *,
    pack: dict[str, Any] | None = None,
) -> list[CapabilityResult]:
    pack = pack or _load_pack()
    dim = int(pack["pn_dim"])
    X, y_fam, sessions = _build_xy(rows, dim=dim)
    assert len(X) == len(rows)
    W_pn, W, k = pack["W_pn_kc"], pack["W_kc_mbon"], pack["k_winners"]
    sealed_m = np.asarray(mask_for_bucket(sessions, split, "sealed"), dtype=bool)
    train_m = np.asarray(mask_for_bucket(sessions, split, "train"), dtype=bool)
    if not sealed_m.any():
        return []

    logits = _kc_codes(X, W_pn, k) @ W
    probs = _softmax_rows(logits)
    pred = logits.argmax(1)
    results: list[CapabilityResult] = []

    # next_action_family multiclass on sealed
    Xs, ys = X[sealed_m], y_fam[sealed_m]
    preds = pred[sealed_m]
    acc = float((preds == ys).mean())
    ridge = ridge_acc(X[train_m], y_fam[train_m], Xs, ys) if train_m.any() else None
    maj = float(np.bincount(ys, minlength=len(FAMILIES)).max() / max(len(ys), 1))
    n_sess = len({sessions[i] for i in range(len(sessions)) if sealed_m[i]})
    results.append(
        CapabilityResult(
            model="mb_gen0_hash64",
            capability_id="next_action_family",
            evidence_level="L0_imitation",
            n=int(sealed_m.sum()),
            n_sessions=n_sess,
            success=acc,
            source=pack["path"],
            transfer="session_sealed",
            inference=False,
        )
    )
    results.append(
        CapabilityResult(
            model="ridge",
            capability_id="next_action_family",
            evidence_level="L0_imitation",
            n=int(sealed_m.sum()),
            n_sessions=n_sess,
            success=ridge,
            source="train_fit_sealed_eval",
            transfer="session_sealed",
            inference=False,
        )
    )
    results.append(
        CapabilityResult(
            model="majority",
            capability_id="next_action_family",
            evidence_level="L0_imitation",
            n=int(sealed_m.sum()),
            n_sessions=n_sess,
            success=maj,
            source="sealed_prior",
            transfer="session_sealed",
            inference=False,
        )
    )

    # per-family pred precision on sealed (capability fingerprint cells)
    for i, name in enumerate(FAMILIES):
        mp = preds == i
        if not mp.any():
            continue
        prec = float((ys[mp] == i).mean())
        results.append(
            CapabilityResult(
                model="mb_gen0_hash64",
                capability_id=f"next_action_pred_{name}",
                evidence_level="L0_imitation",
                n=int(mp.sum()),
                n_sessions=n_sess,
                precision=prec,
                success=prec,
                source=pack["path"],
                transfer="session_sealed",
                inference=False,
            )
        )

    # binary niches
    labels = _proxy_labels(rows)
    del_i = FAM_I["DELEGATE"]

    def _binary(
        cap_id: str,
        y_all: np.ndarray,
        score: np.ndarray,
        *,
        pred_positive: np.ndarray | None = None,
    ) -> None:
        m = sealed_m & (y_all >= 0)
        if not m.any():
            return
        y = y_all[m]
        sc = score[m]
        n_s = len({sessions[i] for i in range(len(sessions)) if m[i]})
        base = float(y.mean())
        maj_acc = float(max(base, 1.0 - base))
        # score-ranked coverage@95 for positive class when predicting positive via threshold
        # Use model positive = pred_positive if given else sc >= 0.5
        if pred_positive is not None:
            pr = pred_positive[m].astype(bool)
        else:
            pr = sc >= 0.5
        prec = float(y[pr].mean()) if pr.any() else None
        n_pos = int(pr.sum())
        # margin-like: rank by score, precision of positive labels among top-k predicted
        order = np.argsort(-sc)
        # treat "predict yes" as taking top-k by score; use coverage_at_precision with y_pred=1 for all in prefix via mask
        # simpler: when pr, precision; also report base
        results.append(
            CapabilityResult(
                model="mb_gen0_hash64",
                capability_id=cap_id,
                evidence_level="L0_imitation",
                n=n_pos if n_pos else int(m.sum()),
                n_sessions=n_s,
                success=float((pr == y).mean()) if n_pos or True else None,
                precision=prec,
                source=pack["path"],
                transfer="session_sealed",
                inference=False,
                cost_note=f"base_rate={base:.4f}; majority_acc={maj_acc:.4f}; n_trigger={int(m.sum())}",
            )
        )
        results.append(
            CapabilityResult(
                model="majority",
                capability_id=cap_id,
                evidence_level="L0_imitation",
                n=int(m.sum()),
                n_sessions=n_s,
                success=maj_acc,
                source="sealed_prior",
                transfer="session_sealed",
                inference=False,
            )
        )
        # coverage@95 on positive predictions ranked by score
        # Build y_true/y_pred for positive class: y_pred always 1 along score order among all triggers
        y_pred = np.ones(int(m.sum()), dtype=np.int32)
        c95 = coverage_at_precision(y, y_pred, sc, floor=0.95)
        results.append(
            CapabilityResult(
                model="mb_gen0_hash64",
                capability_id=cap_id,
                evidence_level="L0_imitation",
                n=int(c95.get("n") or 0),
                n_sessions=n_s,
                coverage_at_95=float(c95.get("coverage") or 0.0),
                precision=c95.get("precision"),
                source="score_ranked_positive_prefix",
                transfer="session_sealed",
                inference=False,
                cost_note="coverage_at_95 among trigger rows ranked by model score",
            )
        )

    _binary(
        "delegate_gating",
        labels["delegate_gating"],
        probs[:, del_i],
        pred_positive=(pred == del_i),
    )
    _binary(
        "needs_verification",
        labels["needs_verification"],
        probs[:, FAM_I["EXECUTE"]] + probs[:, FAM_I["VERIFY"]],
    )
    _binary(
        "retry_execute",
        labels["retry_execute"],
        probs[:, FAM_I["EXECUTE"]],
        pred_positive=(pred == FAM_I["EXECUTE"]),
    )
    _binary(
        "continue_same_family",
        labels["continue_same_family"],
        # score: p(predicted family) if matches stay signal — use max prob as confidence of any stay
        probs.max(axis=1),
        pred_positive=np.array(
            [
                (rows[i].get("prev") or [None])[-1] == FAMILIES[int(pred[i])]
                if (rows[i].get("prev") or [None])[-1] in FAM_I
                else False
                for i in range(len(rows))
            ]
        ),
    )
    return results


def build_cards(stats: dict[str, Any], results: list[CapabilityResult]) -> list[CapabilityCard]:
    days = float(stats.get("span_days") or 1.0)
    fam = stats.get("family") or {}
    # session counts for proxies
    cards: list[CapabilityCard] = []
    by_cap: dict[str, list[CapabilityResult]] = defaultdict(list)
    for r in results:
        by_cap[r.capability_id].append(r)

    # measured event frequencies from stats / known proxies
    freq_map = {
        "delegate_gating": int(fam.get("DELEGATE") or 0),
        "next_action_family": int(stats.get("n_events") or 0),
        "needs_verification": None,  # filled below from evidence notes
        "retry_execute": None,
        "continue_same_family": None,
        "read_vs_edit": int(fam.get("EDIT") or 0),
        "recovery_action": int((stats.get("tier") or {}).get("fail") or 0),
        "skill_routing": 0,
        "context_file_relevance": 0,
        "task_complete": int(fam.get("RESPOND") or 0),
    }

    for seed in SEED_CARDS:
        cid = seed["id"]
        notes = []
        for r in by_cap.get(cid, []):
            if r.model.startswith("mb") and r.precision is not None:
                notes.append(
                    f"L0 {r.model} precision={r.precision:.4f} n={r.n} sealed_sessions={r.n_sessions} ({r.source})"
                )
            if r.model.startswith("mb") and r.success is not None and r.coverage_at_95 is None:
                notes.append(
                    f"L0 {r.model} success/acc={r.success:.4f} n={r.n} ({r.cost_note or r.source})"
                )
            if r.coverage_at_95 is not None:
                notes.append(
                    f"L0 coverage@95={r.coverage_at_95:.4f} prec={r.precision} n={r.n}"
                )
        for r in by_cap.get(f"next_action_pred_DELEGATE", []):
            if cid == "delegate_gating" and r.precision is not None:
                notes.append(
                    f"SEALED when MB predicts DELEGATE: precision={r.precision:.4f} n_pred={r.n} "
                    f"(session-sealed; not L2/L3)"
                )
        gap = bool(seed.pop("instrumentation_gap", False)) if "instrumentation_gap" in seed else False
        # restore seed dict without mutating SEED_CARDS permanently
        seed = dict(seed)
        seed.pop("instrumentation_gap", None)
        fe = freq_map.get(cid) or 0
        # annotate gaps
        if gap:
            notes.append("INSTRUMENTATION_GAP: needs path/outcome join not present in next_action.jsonl alone")
        if cid == "next_action_family":
            for r in by_cap.get(cid, []):
                if r.model == "mb_gen0_hash64" and r.success is not None:
                    notes.append(
                        f"SEALED multiclass MB acc={r.success:.4f} vs ridge/majority in sibling results; "
                        f"DEV confirm 0.572 was search-contaminated — treat as L0 screening only"
                    )
        card = CapabilityCard(
            **{k: v for k, v in seed.items() if k in CapabilityCard.__dataclass_fields__},
            frequency_events=int(fe),
            frequency_sessions=int(stats.get("n_sessions") or 0) if fe else 0,
            events_per_day=float(fe) / days if fe else 0.0,
            evidence_notes=notes,
            current_cost_note="frontier/Jev call when escalated; local MB <<1ms if absorbed",
        )
        if gap:
            card.label_quality = "low"
        cards.append(card)
    return cards


def build_specs(cards: list[CapabilityCard], split) -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []
    split_desc = {
        "train": f"oldest sessions n={len(split.train_sessions)} events={split.n_train_events}",
        "dev": f"mid-recent sessions n={len(split.dev_sessions)} events={split.n_dev_events} (search may inspect)",
        "sealed": f"newest held-out sessions n={len(split.sealed_sessions)} events={split.n_sealed_events} (evolver-blind)",
        "future": "live traffic after model freeze (not yet accumulated)",
    }
    for c in cards:
        specs.append(
            BenchmarkSpec(
                capability_id=c.id,
                split=split_desc,
                label_rule=c.verifier,
                success_criterion=(
                    "On SEALED sessions: specialist beats majority+ridge (same features) on primary metric; "
                    "for gating tasks also report coverage@≥95% precision."
                ),
                failure_criterion="No sealed lift vs simplest applicable control, or transfer collapses across harness/repo.",
                notes=c.description,
                controls=list(DEFAULT_CONTROLS),
            )
        )
    return specs


def write_deliverables(
    *,
    stats: dict[str, Any],
    cards: list[CapabilityCard],
    specs: list[BenchmarkSpec],
    results: list[CapabilityResult],
    split,
    out_dir: Path | None = None,
) -> dict[str, str]:
    out_dir = out_dir or RESEARCH
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    cards_path = out_dir / "capability-cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as fh:
        for c in cards:
            fh.write(json.dumps(c.to_dict()) + "\n")
    paths["cards"] = str(cards_path)

    specs_path = out_dir / "benchmark-specs.jsonl"
    with specs_path.open("w", encoding="utf-8") as fh:
        for s in specs:
            fh.write(json.dumps(s.to_dict()) + "\n")
    paths["specs"] = str(specs_path)

    results_path = out_dir / "capability-results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict()) + "\n")
    paths["results"] = str(results_path)

    split_path = out_dir / "session-split.json"
    split_path.write_text(json.dumps(split.to_dict(), indent=2) + "\n")
    paths["split"] = str(split_path)

    # top3 from evidence
    top3 = _select_top3(cards, results)
    top3_path = out_dir / "top3-plan.md"
    top3_path.write_text(_render_top3(top3, stats, split) + "\n")
    paths["top3"] = str(top3_path)

    gaps_path = out_dir / "instrumentation-gaps.md"
    gaps_path.write_text(_render_gaps(cards, stats) + "\n")
    paths["gaps"] = str(gaps_path)

    atlas_path = out_dir / "capability-atlas.md"
    atlas_path.write_text(_render_atlas(stats, cards, results, split, top3) + "\n")
    paths["atlas"] = str(atlas_path)

    meta = {
        "schema": "flyforge.capability_mine.v1",
        "ts": time.time(),
        "paths": paths,
        "stats_summary": {
            "n_events": stats.get("n_events"),
            "n_sessions": stats.get("n_sessions"),
            "span_days": stats.get("span_days"),
        },
        "n_cards": len(cards),
        "n_results": len(results),
    }
    meta_path = out_dir / "mine-meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    paths["meta"] = str(meta_path)
    return paths


def _select_top3(
    cards: list[CapabilityCard], results: list[CapabilityResult]
) -> list[CapabilityCard]:
    """Prefer measured L0 lift + verifiability + frequency; skip pure gaps if better exist."""
    score: dict[str, float] = {}
    for c in cards:
        s = 0.0
        s += math.log1p(c.frequency_events) * 0.5
        if c.failure_cost == "high":
            s += 2.0
        elif c.failure_cost == "medium":
            s += 1.0
        if c.label_quality == "high":
            s += 2.0
        elif c.label_quality == "medium":
            s += 1.0
        # evidence bonuses
        for note in c.evidence_notes:
            if "precision=0.98" in note or "precision=0.9" in note:
                s += 5.0
            if "SEALED when MB predicts DELEGATE" in note:
                s += 8.0
            if "INSTRUMENTATION_GAP" in note:
                s -= 3.0
            if "coverage@95" in note:
                s += 1.5
        if c.id == "recovery_action":
            s += 3.0  # existing P0 gym
        if c.id == "delegate_gating":
            s += 4.0
        if c.id == "needs_verification":
            s += 3.5
        if c.id == "context_file_relevance":
            s += 2.5  # strategic, but gap
        score[c.id] = s
    ranked = sorted(cards, key=lambda c: -score.get(c.id, 0.0))
    # force diversity: take top with preference to implementable
    picked: list[CapabilityCard] = []
    for c in ranked:
        if len(picked) >= 3:
            break
        picked.append(c)
    return picked


def _render_top3(top3: list[CapabilityCard], stats: dict[str, Any], split) -> str:
    lines = [
        "# Top 3 capability experiments",
        "",
        f"Corpus: {stats.get('n_events')} events, {stats.get('n_sessions')} sessions, "
        f"~{stats.get('span_days'):.1f} days.",
        f"Sealed battery: {len(split.sealed_sessions)} sessions / {split.n_sealed_events} events "
        f"(evolver-blind). Prior next-action `confirm` row-split is **DEV only**.",
        "",
    ]
    for i, c in enumerate(top3, 1):
        lines += [
            f"## {i}. `{c.id}`",
            "",
            f"**Why:** {c.description}",
            f"**Stage/trigger:** {c.stage} / {c.trigger}",
            f"**Output:** {c.output_contract}",
            f"**Verifier:** {c.verifier}",
            f"**Failure cost:** {c.failure_cost}; **label quality:** {c.label_quality}",
            f"**Frequency (events):** {c.frequency_events} (~{c.events_per_day:.1f}/day)",
            "",
            "### Reuse",
            "- `~/.z0int/episodes/next_action.jsonl` (+ gen-0 pack for L0 fingerprint)",
            "- Session split: `~/.z0int/research/session-split.json`",
            "- Controls: majority, rule, ridge, MB, OpenJev, Jev",
            "",
            "### <1-day change",
            "- Binary/multiclass label column from existing `family`+`prev`",
            "- Eval on sealed sessions only; freeze thresholds on dev",
            "- Metric: accuracy + coverage@95% precision + calibration",
            "",
            "### Success / failure",
            f"- Success: {c.verifier[:80]}… and sealed lift vs ridge+majority",
            "- Failure: no sealed lift, or precision@coverage collapses on future live stream",
            "",
            "### What would change architecture",
            "- Strong sealed L0/L2 → promote specialist shadow route for this card only",
            "- Needs recurrence/temporal depth → consider fly/GRU (still not MaleCNS full graph)",
            "- Needs paths/screenshots → retrieval/vision specialist, not bigger next-action MB",
            "",
            "### Evidence notes",
        ]
        for n in c.evidence_notes or ["(none yet)"]:
            lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines)


def _render_gaps(cards: list[CapabilityCard], stats: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Instrumentation gaps (outcome-backed evaluation)",
            "",
            "## Blocking L2/L3",
            "",
            "1. **Tool result join** — episodes store family/tool/tier but not structured exit codes, "
            "test pass/fail, or patch revert. `tier=gold` means tool `success:true`, not optimal action.",
            "2. **Path/span ground truth** — no first-class record of files ultimately edited vs merely read "
            "→ blocks `context_file_relevance` L2.",
            "3. **User correction signals** — rare in compiled episodes; need OMP/Hermes explicit correction events.",
            "4. **Closed-loop canary hooks** — flyforge-jev is log-only; no harness switch to let specialist "
            "act with fallback measurement (L3).",
            "5. **Live stream thin** — `~/.z0int/stream` still low volume; multi-session `session_id` often null.",
            "6. **Jev soft probs** — shadow rows often lack full probability vectors (skill names ≠ families).",
            "7. **Energy ledger** — joules/verified-success not wired (NVML snapshots exist in GPU evolve only).",
            "",
            "## What we can measure now (L0/L1 partial)",
            "",
            f"- Historical next-family imitation on {stats.get('n_events')} rows / {stats.get('n_sessions')} sessions",
            "- Session-sealed transfer for next-action / delegate / verify-proxy / retry-proxy",
            "- Cascade coverage@precision on DEV (contaminated) vs SEALED (honest)",
            "- Recovery gym already exists in evolution-lab P0 (typed state → bounded action)",
            "",
            "## Minimal joins to unlock L2",
            "",
            "```text",
            "trace_id → tool_result.{ok,exit,stderr_hash}",
            "trace_id → test_result.{passed,failed,name_hash}",
            "trace_id → files_touched.[path_hash]",
            "trace_id → user_correction.bool",
            "```",
            "",
            "Keep raw text private; store hashes + booleans in `outcome_gold.jsonl`.",
            "",
            "## Cards marked gap-heavy",
            "",
        ]
        + [f"- `{c.id}` — {c.description}" for c in cards if any("INSTRUMENTATION_GAP" in n for n in c.evidence_notes)]
        + [""]
    )


def _render_atlas(
    stats: dict[str, Any],
    cards: list[CapabilityCard],
    results: list[CapabilityResult],
    split,
    top3: list[CapabilityCard],
) -> str:
    del_pred = next(
        (
            r
            for r in results
            if r.capability_id == "next_action_pred_DELEGATE" and r.precision is not None
        ),
        None,
    )
    na_mb = next(
        (
            r
            for r in results
            if r.capability_id == "next_action_family" and r.model.startswith("mb")
        ),
        None,
    )
    na_ridge = next(
        (r for r in results if r.capability_id == "next_action_family" and r.model == "ridge"),
        None,
    )
    na_maj = next(
        (r for r in results if r.capability_id == "next_action_family" and r.model == "majority"),
        None,
    )
    lines = [
        "# z0int Capability Atlas",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')} — discovery only; not a product leaderboard._",
        "",
        "## Thesis",
        "",
        "Do **not** optimize “next-action accuracy” as the task. Per Frontier KB "
        "(`experiments-are-the-population`, `harness-evolver`, `specialist-subagent-handoff`):",
        "",
        "- experiments (TASK × DATA × MODEL × POLICY) are the population;",
        "- specialists need typed outputs + external verifiers;",
        "- credit only on an **evolver-blind sealed battery** (session-held-out);",
        "- score joules/tokens/seconds per **verified** success, including failures/fallbacks.",
        "",
        "The mushroom body is a **worker/specialist candidate**, not a universal orchestrator.",
        "",
        "## What 0.572 actually meant",
        "",
        "| Claim | Status |",
        "| --- | --- |",
        "| MB L0 next-family on **DEV** row-split | ~0.572 vs ridge ~0.363 vs maj ~0.353 |",
        "| Final sealed benchmark | **No** — confirm was used for pop selection + recipe search |",
        "| Task success | **Not measured** |",
        "| Gold = optimal action | **False** — gold = tool success flag |",
        "",
        f"Session-**sealed** reprobe (gen-0 hash64 pack): "
        f"MB={getattr(na_mb, 'success', None)} ridge={getattr(na_ridge, 'success', None)} "
        f"majority={getattr(na_maj, 'success', None)} "
        f"(n={getattr(na_mb, 'n', None)} events, {getattr(na_mb, 'n_sessions', None)} sessions).",
        "",
        "Note: sealed majority can dominate if newest sessions are class-skewed "
        "(here often DELEGATE-heavy). Still report MB vs ridge on **same** features.",
        "",
        "## Surprising niche: DELEGATE gating",
        "",
        (
            f"When MB **predicts** DELEGATE on sealed sessions: precision="
            f"{del_pred.precision:.4f} n_pred={del_pred.n}."
            if del_pred and del_pred.precision is not None
            else "DELEGATE pred precision: see capability-results.jsonl"
        ),
        "",
        "Evidence level: **L0 imitation only**. Not verified delegation quality (L2/L3).",
        "But this is exactly a capability fingerprint cell: mediocre global, sharp narrow yes-slice.",
        "",
        "## Corpus",
        "",
        f"- events: {stats.get('n_events')}",
        f"- sessions: {stats.get('n_sessions')} (median len {stats.get('median_session_len')})",
        f"- span_days: {stats.get('span_days'):.2f}",
        f"- events/day: {stats.get('events_per_day'):.1f}",
        f"- tier: {stats.get('tier')}",
        f"- family mass: {stats.get('family')}",
        f"- split: train_sess={len(split.train_sessions)} dev={len(split.dev_sessions)} "
        f"sealed={len(split.sealed_sessions)} "
        f"(events {split.n_train_events}/{split.n_dev_events}/{split.n_sealed_events})",
        "",
        "## Capability cards (summary)",
        "",
        "| id | stage | events | fail_cost | label_q | notes |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for c in cards:
        note = (c.evidence_notes[0][:60] + "…") if c.evidence_notes else ""
        lines.append(
            f"| `{c.id}` | {c.stage} | {c.frequency_events} | {c.failure_cost} | {c.label_quality} | {note} |"
        )
    lines += [
        "",
        "## Fingerprint matrix (partial, L0 sealed)",
        "",
        "| capability | MB | ridge | majority | evidence |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    # pivot a few
    for cap in (
        "next_action_family",
        "delegate_gating",
        "needs_verification",
        "retry_execute",
        "continue_same_family",
        "next_action_pred_DELEGATE",
        "next_action_pred_EXECUTE",
    ):
        cells = {r.model: r for r in results if r.capability_id == cap}
        if not cells:
            continue
        def _fmt(m: str) -> str:
            r = cells.get(m)
            if not r:
                return "?"
            if r.precision is not None and "pred_" in cap:
                return f"{r.precision:.3f}(n={r.n})"
            if r.success is not None:
                return f"{r.success:.3f}"
            if r.precision is not None:
                return f"p={r.precision:.3f}"
            return "?"
        lines.append(
            f"| `{cap}` | {_fmt('mb_gen0_hash64')} | {_fmt('ridge')} | {_fmt('majority')} | L0 sealed |"
        )
    lines += [
        "",
        "## Top 3 next experiments",
        "",
    ]
    for c in top3:
        lines.append(f"1. **`{c.id}`** — {c.description}")
    lines += [
        "",
        "Details: `top3-plan.md`. Gaps: `instrumentation-gaps.md`.",
        "",
        "## Product metrics (definitions)",
        "",
        "For each capability after L2 instrumentation:",
        "",
        r"- safe_offload/day = frequency/day × coverage_at_required_precision",
        r"- tokens_saved/day = safe_offload/day × (baseline_tokens − specialist_tokens)",
        r"- joules/verified_success = whole_loop_energy_including_fail_fallback / verified_successes",
        "",
        "Do not collapse to one Opportunity Score; keep Pareto dimensions "
        "(frequency, removable cost, verifiability, boundedness, failure cost, transfer).",
        "",
        "## Non-goals (until atlas says otherwise)",
        "",
        "- MaleCNS full graph / FlyGym / broad 5B SFT / AODL-critical-path",
        "- Treating DEV confirm 0.572 as a sealed claim",
        "- Paying Jev to label every row",
        "",
        "## Sources",
        "",
        "- `~/.z0int/episodes/next_action.jsonl`",
        "- gen-0 pack `runs/gpu-evolve/gen0_champion_backup.npz` (hash64)",
        "- frontier-kb: experiments-are-the-population, harness-evolver, p0-recovery, "
        "joules-per-verified-success, specialist-subagent-handoff, small-models-are-workers-or-specialists",
        "",
    ]
    return "\n".join(lines)


def run_capability_mine(
    *,
    episodes: Path | None = None,
    out_dir: Path | None = None,
    sealed_frac: float = 0.10,
    dev_frac: float = 0.10,
) -> dict[str, Any]:
    path = episodes or EPISODES
    rows = load_episode_rows(path)
    # filter to labeled families for alignment with XY
    rows = [r for r in rows if r.get("family") in FAM_I]
    stats = mine_corpus_stats(rows)
    split = make_session_split(rows, sealed_frac=sealed_frac, dev_frac=dev_frac)
    results = evaluate_l0_niches(rows, split)
    cards = build_cards(stats, results)
    # fill proxy frequencies from labels
    labels = _proxy_labels(rows)
    days = float(stats["span_days"])
    for c in cards:
        if c.id in labels:
            m = labels[c.id] >= 0
            c.frequency_events = int(m.sum())
            c.events_per_day = c.frequency_events / days
            c.frequency_sessions = len(
                {str(rows[i].get("session")) for i in range(len(rows)) if m[i]}
            )
    specs = build_specs(cards, split)
    paths = write_deliverables(
        stats=stats, cards=cards, specs=specs, results=results, split=split, out_dir=out_dir
    )
    return {
        "schema": "flyforge.capability_mine_report.v1",
        "stats": {
            "n_events": stats["n_events"],
            "n_sessions": stats["n_sessions"],
            "span_days": stats["span_days"],
            "family": stats["family"],
            "tier": stats["tier"],
        },
        "split": {
            "n_train_sessions": len(split.train_sessions),
            "n_dev_sessions": len(split.dev_sessions),
            "n_sealed_sessions": len(split.sealed_sessions),
            "n_train_events": split.n_train_events,
            "n_dev_events": split.n_dev_events,
            "n_sealed_events": split.n_sealed_events,
        },
        "n_cards": len(cards),
        "n_specs": len(specs),
        "n_results": len(results),
        "paths": paths,
        "top3": [c.id for c in _select_top3(cards, results)],
    }
