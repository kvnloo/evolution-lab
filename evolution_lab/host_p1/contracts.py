"""Immutable, synthetic-only state, intervention, and outcome contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

SCHEMA_VERSION = "host_p1.v1"
schema_version = SCHEMA_VERSION
TRUTH_TIER = "synthetic"
ACTUATION_AUTHORITY = False

_FRACTIONS = (
    "mem_used_frac",
    "swap_used_frac",
    "zram_used_frac",
    "psi_mem_some_avg10",
    "ui_cgroup_mem_frac",
)


def _seconds(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(name + " must be a nonnegative integer" if not positive else name + " must be a positive integer")


def _finite_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise ValueError(name + " must be a finite number")


@dataclass(frozen=True)
class StateFrame:
    t: int
    mem_used_frac: float
    swap_used_frac: float
    zram_used_frac: float
    psi_mem_some_avg10: float
    n_procs: int
    ui_cgroup_mem_frac: float
    truth_tier: str = field(default=TRUTH_TIER, init=False)

    def __post_init__(self) -> None:
        _seconds(self.t, "t")
        _seconds(self.n_procs, "n_procs")
        for name in _FRACTIONS:
            value = getattr(self, name)
            _finite_real(value, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(name + " must be in [0, 1]")


@dataclass(frozen=True)
class Intervention:
    t: int
    knob: str
    intended_value: float
    effective_value: float
    group_id: str
    truth_tier: str = field(default=TRUTH_TIER, init=False)

    def __post_init__(self) -> None:
        _seconds(self.t, "t")
        if not isinstance(self.knob, str) or not self.knob:
            raise ValueError("knob must be a nonempty string")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a nonempty string")
        _finite_real(self.intended_value, "intended_value")
        _finite_real(self.effective_value, "effective_value")

    @property
    def effective_matches_intent(self) -> bool:
        return self.effective_value == self.intended_value or (
            abs(self.effective_value - self.intended_value) < 1e-9
        )


@dataclass(frozen=True)
class Outcome:
    t: int
    oom_kill_in_window: bool
    ui_stall_in_window: bool
    window_s: int
    truth_tier: str = field(default=TRUTH_TIER, init=False)

    def __post_init__(self) -> None:
        _seconds(self.t, "t")
        _seconds(self.window_s, "window_s", positive=True)
        if type(self.oom_kill_in_window) is not bool:
            raise ValueError("oom_kill_in_window must be bool")
        if type(self.ui_stall_in_window) is not bool:
            raise ValueError("ui_stall_in_window must be bool")
