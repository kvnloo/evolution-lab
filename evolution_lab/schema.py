"""Experiment genome: the fundamental object of Evolution Lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any
import json
from pathlib import Path

from .targets import PRODUCT_TARGET

ACTIONS = ("retry", "restart_sandbox", "escalate", "noop", "page_human")
FAMILIES = (
    "rule",
    "direct_input",
    "mlp",
    "gru",
    "fixed_reservoir",
    "rewired_reservoir",
    "local_plasticity",
    "fly_connectome",
    "hybrid",
)
BACKENDS = ("local_numpy", "fly_sim", "tinker_sft", "tinker_rl", "external_eval")
ROLES = ("explorer", "exploiter", "skeptic", "replicator", "distiller", "neuroscience")
LEARNING_MODES = ("fixed", "ridge_readout", "sft", "rl", "local_plasticity", "hybrid")
PARAM_BUCKETS = ("<100K", "100K-1M", "1M-10M", "10M-100M", ">100M")

SECRET_FIELD_NAMES = frozenset(
    {"credential", "password", "secret", "token", "api_key", "bws_payload"}
)


class GenomeError(ValueError):
    pass


@dataclass
class Architecture:
    family: str = "mlp"
    hidden: int = 32
    history: int = 8
    sparsity: float = 0.05
    spectral_radius: float = 0.9
    trainable: str = "readout"
    topology: str = "synthetic"


@dataclass
class Curriculum:
    task: str = "hermes_recovery"
    include_secrets: bool = False
    delayed_cue: bool = False
    strobe_drop: float = 0.0
    ood_split: str = "unseen_failure"


@dataclass
class Training:
    algorithm: str = "ridge_readout"
    seed: int = 0
    l2: float = 1e-2
    budget_steps: int = 1
    teacher: str = "teacher_rule"


@dataclass
class Evaluation:
    seeds: int = 1
    split: str = "confirm"
    environments: tuple[str, ...] = ("iid", "ood")


@dataclass
class ExperimentGenome:
    id: str
    lineage: str
    hypothesis: str
    parents: tuple[str, ...] = ()
    role: str = "exploiter"
    backend: str = "local_numpy"
    architecture: Architecture = field(default_factory=Architecture)
    curriculum: Curriculum = field(default_factory=Curriculum)
    training: Training = field(default_factory=Training)
    evaluation: Evaluation = field(default_factory=Evaluation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.architecture.family not in FAMILIES:
            raise GenomeError(f"unknown family {self.architecture.family}")
        if self.backend not in BACKENDS:
            raise GenomeError(f"unknown backend {self.backend}")
        if self.role not in ROLES:
            raise GenomeError(f"unknown role {self.role}")
        if self.curriculum.include_secrets:
            raise GenomeError("secret-bearing observations fail closed at L0")
        if not 0 <= self.curriculum.strobe_drop < 1:
            raise GenomeError("strobe_drop must be in [0, 1)")
        if self.architecture.hidden < 1 or self.architecture.history < 1:
            raise GenomeError("hidden and history must be positive")
        if self.architecture.family == "rule" and self.backend not in {"local_numpy", "external_eval"}:
            raise GenomeError("rule family is local")

    def with_seed(self, seed: int) -> "ExperimentGenome":
        return replace(self, training=replace(self.training, seed=seed))


def param_bucket(n_params: int) -> str:
    if n_params < 100_000:
        return "<100K"
    if n_params < 1_000_000:
        return "100K-1M"
    if n_params < 10_000_000:
        return "1M-10M"
    if n_params < 100_000_000:
        return "10M-100M"
    return ">100M"


def load_genome(path: Path) -> ExperimentGenome:
    data = json.loads(path.read_text())
    return genome_from_dict(data)


def genome_from_dict(data: dict[str, Any]) -> ExperimentGenome:
    arch = Architecture(**data.get("architecture", {}))
    cur = Curriculum(**data.get("curriculum", {}))
    tr = Training(**data.get("training", {}))
    ev_raw = dict(data.get("evaluation", {}))
    if "environments" in ev_raw:
        ev_raw["environments"] = tuple(ev_raw["environments"])
    ev = Evaluation(**ev_raw)
    g = ExperimentGenome(
        id=data["id"],
        lineage=data["lineage"],
        hypothesis=data["hypothesis"],
        parents=tuple(data.get("parents") or ()),
        role=data.get("role", "exploiter"),
        backend=data.get("backend", "local_numpy"),
        architecture=arch,
        curriculum=cur,
        training=tr,
        evaluation=ev,
    )
    g.validate()
    return g


def observation_is_sanitized(fields: dict[str, Any]) -> bool:
    return SECRET_FIELD_NAMES.isdisjoint(fields)


def options_carry_secrets(options: dict[str, Any] | None) -> bool:
    """True if reset/step options include the L0 secret flag or a secret-bearing key."""
    if not options:
        return False
    if options.get("include_secrets"):
        return True
    return not SECRET_FIELD_NAMES.isdisjoint(options)
