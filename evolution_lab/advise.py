"""Sanitized Hermes observation → recovery action.

The useful object is a cheap specialist on
{retry, restart_sandbox, escalate, noop, page_human}.
Secrets fail closed. This is not a DEX trader and not a Qwen replacement.
"""

from __future__ import annotations

from typing import Any
import json

import numpy as np

from .schema import ACTIONS, GenomeError, options_carry_secrets
from .task import Episode, N_ACTIONS, _frame, teacher_action


def _last_action_id(value: Any) -> int:
    if value is None:
        return ACTIONS.index("noop")
    if isinstance(value, str):
        if value not in ACTIONS:
            raise GenomeError(f"unknown action {value!r}")
        return ACTIONS.index(value)
    idx = int(value)
    if not 0 <= idx < N_ACTIONS:
        raise GenomeError(f"action {idx} outside Discrete({N_ACTIONS})")
    return idx


def fields_to_frame(fields: dict[str, Any]) -> np.ndarray:
    if options_carry_secrets(fields):
        raise GenomeError("secret-bearing observations fail closed at L0")
    return _frame(
        sandbox_alive=float(fields.get("sandbox_alive", 1.0)),
        retry=int(fields.get("retry", 0)),
        budget=float(fields.get("budget", 1.0)),
        elapsed=float(fields.get("elapsed", 1.0)),
        transient=float(fields.get("transient", 0.0)),
        hard=float(fields.get("hard", 0.0)),
        unfamiliar=float(fields.get("unfamiliar", 0.0)),
        policy_hit=float(fields.get("policy_hit", 0.0)),
        already_ok=float(fields.get("already_ok", 0.0)),
        cue=float(fields.get("cue", 0.0)),
        last_action=_last_action_id(fields.get("last_action")),
    )


def episode_from_fields(fields: dict[str, Any], *, history: int = 8) -> Episode:
    """Pad a last-step observation into a history window. Cue may live at t=0."""
    last = fields_to_frame(fields)
    cue0 = float(fields.get("cue_seen", 0.0))
    frames = []
    for t in range(history):
        fr = last.copy()
        fr[3] = t / max(history - 1, 1)
        fr[9] = cue0 if t == 0 else 0.0
        frames.append(fr)
    stacked = np.stack(frames)
    labels = np.zeros(history, dtype=np.int64)
    if cue0 > 0.5:
        labels[-1] = ACTIONS.index("escalate")
    else:
        labels[-1] = teacher_action(last)
    return Episode(stacked, labels, env="iid")


def advise(fields: dict[str, Any], *, family: str = "rule") -> dict[str, Any]:
    """Return a recovery action for a sanitized observation dict."""
    if family == "rule":
        frame = fields_to_frame(fields)
        y = teacher_action(frame)
        if float(fields.get("cue_seen", 0.0)) > 0.5:
            y = ACTIONS.index("escalate")
        return {
            "action": ACTIONS[int(y)],
            "family": "rule",
            "source": "teacher_rule",
            "actions": list(ACTIONS),
        }
    if family != "local_plasticity":
        raise GenomeError(
            f"advise supports family 'rule' or 'local_plasticity', not {family!r}"
        )
    from .engine import seed_genomes
    from .models import fit_student
    from .task import build_task

    genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
    data = build_task(genome, n_train=64, n_val=8, n_confirm=8, n_ood=4)
    student = fit_student(genome, data.train)
    ep = episode_from_fields(fields, history=genome.architecture.history)
    pred = int(student.predict_fn([ep])[0])
    return {
        "action": ACTIONS[pred],
        "family": "local_plasticity",
        "source": "kc_to_mbon_local_plasticity",
        "encoder": student.extras.get("encoder"),
        "n_params": student.n_params,
        "actions": list(ACTIONS),
        "note": "PN drive is flatten+last+maxpool Hermes history, not the compound eye.",
    }


def advise_json(payload: str, *, family: str = "rule") -> str:
    fields = json.loads(payload)
    if not isinstance(fields, dict):
        raise GenomeError("advise JSON must be an object")
    return json.dumps(advise(fields, family=family), indent=2) + "\n"
