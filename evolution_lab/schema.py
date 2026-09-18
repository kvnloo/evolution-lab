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
    "jev_tiny",
    "jev_hf_head",
)
BACKENDS = ("local_numpy", "local_jax", "fly_sim", "tinker_sft", "tinker_rl", "external_eval", "openjev")
JEV_FAMILIES = frozenset({"jev_tiny", "jev_hf_head"})
JEV_TASKS = frozenset({"jev_synthetic", "hermes_as_jev"})
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
    k_winners: int = 0  # 0 = 10% of hidden


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
    jev_rank: int = 64
    jev_epochs: int = 8
    jev_lr: float = 2e-3
    jev_hf_model: str = "Qwen/Qwen2.5-0.5B"
    jev_context_tokens: int = 192
    jev_option_tokens: int = 32
    jev_batch_size: int = 64
    jev_device: str = "auto"
    plasticity_lr: float = 0.35
    plasticity_epochs: int = 20


@dataclass
class DataRecipe:
    """Evolvable dataset recipe. Judge/splits/privacy are not in here."""

    n_train: int = 64
    gold_only: bool = False
    source: str = "hermes_recovery"  # hermes_recovery | next_action
    confirm_frac: float = 0.2


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
    recipe: DataRecipe = field(default_factory=DataRecipe)

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
        if self.architecture.family in JEV_FAMILIES:
            if self.backend != "openjev":
                raise GenomeError(f"{self.architecture.family} requires backend openjev")
            if self.curriculum.task not in JEV_TASKS:
                raise GenomeError(f"{self.architecture.family} requires a Jev curriculum task")
        if self.backend == "openjev" and self.architecture.family not in JEV_FAMILIES:
            raise GenomeError("openjev backend is Route A only (jev_tiny, jev_hf_head)")
        if self.backend == "local_jax" and self.architecture.family != "local_plasticity":
            raise GenomeError("local_jax implements local_plasticity only")
        if self.recipe.source not in {"hermes_recovery", "next_action"}:
            raise GenomeError(f"unknown recipe.source {self.recipe.source}")
        if self.recipe.n_train not in {0} and self.recipe.n_train < 8:
            raise GenomeError("recipe.n_train must be 0 (all) or >= 8")
        if not 0 < self.recipe.confirm_frac < 1:
            raise GenomeError("recipe.confirm_frac must be in (0, 1)")

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
    rec = DataRecipe(**(data.get("recipe") or {}))
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
        recipe=rec,
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
