"""Autoresearch selection: hard gates + latency champion compare.

SearchDriver must never edit this module or bench.py (frontier-kb cycle-10 split).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FlyCandidate:
    """One autoresearch experiment: genome + training/bench knobs."""

    genome_id: str
    genome: dict[str, Any]
    dagger_rounds: int = 0
    plasticity_lr: float = 0.35
    plasticity_epochs: int = 20
    k_winners: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlyCandidate":
        return cls(
            genome_id=str(data["genome_id"]),
            genome=dict(data["genome"]),
            dagger_rounds=int(data.get("dagger_rounds") or 0),
            plasticity_lr=float(data.get("plasticity_lr") or 0.35),
            plasticity_epochs=int(data.get("plasticity_epochs") or 20),
            k_winners=int(data.get("k_winners") or 0),
            description=str(data.get("description") or ""),
        )


@dataclass
class BenchMetrics:
    closed_loop_reward: float = 0.0
    confirm_success: float = 0.0
    val_success: float = 0.0
    ood_success: float = 0.0
    violations: float = 0.0
    n_params: int = 0
    fit_train_ms: float = 0.0
    bundle_save_ms: float = 0.0
    advise_cold_ms: float = 0.0
    advise_warm_p50_ms: float = 0.0
    advise_warm_p99_ms: float = 0.0
    closed_loop_ms: float = 0.0
    omp_bridge_ms: float = 0.0
    unit_tests_ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchMetrics":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class BenchResult:
    candidate: FlyCandidate
    metrics: BenchMetrics
    latency_score: float
    gates_pass: bool
    gate_reasons: list[str] = field(default_factory=list)
    status: str = "discard"
    experiment_id: str = ""
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict(),
            "latency_score": self.latency_score,
            "gates_pass": self.gates_pass,
            "gate_reasons": self.gate_reasons,
            "status": self.status,
            "wall_s": self.wall_s,
        }


def load_config(path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    cfg_path = path or (root / "autoresearch" / "config.json")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def latency_score(metrics: BenchMetrics, config: dict[str, Any]) -> float:
    obj = config.get("objective") or {}
    w_advise = float(obj.get("latency_weight_advise_p99_ms", 0.7))
    w_closed = float(obj.get("latency_weight_closed_loop_ms", 0.3))
    return w_advise * metrics.advise_warm_p99_ms + w_closed * metrics.closed_loop_ms


def check_gates(metrics: BenchMetrics, config: dict[str, Any]) -> tuple[bool, list[str]]:
    if metrics.error:
        return False, [f"error: {metrics.error}"]
    if not metrics.unit_tests_ok:
        return False, ["unit_tests_failed"]
    target = config.get("product_target") or {}
    reasons: list[str] = []
    ok = True
    checks = [
        ("closed_loop_reward", metrics.closed_loop_reward, float(target.get("closed_loop_min", 1.0)), ">="),
        ("confirm_success", metrics.confirm_success, float(target.get("confirm_min", 0.95)), ">="),
        ("val_success", metrics.val_success, float(target.get("val_min", 0.95)), ">="),
        ("ood_success", metrics.ood_success, float(target.get("ood_min", 0.85)), ">="),
        ("violations", metrics.violations, float(target.get("extra_violations_max", 0)), "<="),
    ]
    for name, value, bound, op in checks:
        if op == ">=" and value + 1e-9 < bound:
            ok = False
            reasons.append(f"{name}={value:.4f} < {bound}")
        elif op == "<=" and value - 1e-9 > bound:
            ok = False
            reasons.append(f"{name}={value:.4f} > {bound}")
    return ok, reasons


def decide_status(
    result: BenchResult,
    champion: BenchResult | None,
    config: dict[str, Any],
) -> str:
    if not result.gates_pass:
        return "discard"
    if champion is None:
        return "keep"
    eps = float((config.get("objective") or {}).get("improve_epsilon", 0.05))
    threshold = champion.latency_score * (1.0 - eps)
    if result.latency_score < threshold:
        return "keep"
    return "discard"


def champion_path(run_dir: Path) -> Path:
    return run_dir / "champion.json"


def results_tsv_path(run_dir: Path) -> Path:
    return run_dir / "results.tsv"


def load_champion(run_dir: Path) -> BenchResult | None:
    path = champion_path(run_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return BenchResult(
        experiment_id=str(data.get("experiment_id") or ""),
        candidate=FlyCandidate.from_dict(data["candidate"]),
        metrics=BenchMetrics.from_dict(data["metrics"]),
        latency_score=float(data["latency_score"]),
        gates_pass=bool(data.get("gates_pass")),
        gate_reasons=list(data.get("gate_reasons") or []),
        status=str(data.get("status") or "keep"),
        wall_s=float(data.get("wall_s") or 0.0),
    )


def save_champion(run_dir: Path, result: BenchResult) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = champion_path(run_dir)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


RESULTS_HEADER = (
    "experiment_id\tlatency_score\tadvise_warm_p99_ms\tclosed_loop_ms\t"
    "closed_loop_reward\tconfirm_success\tn_params\tstatus\tdescription\n"
)


def append_results_tsv(run_dir: Path, result: BenchResult) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = results_tsv_path(run_dir)
    if not path.is_file():
        path.write_text(RESULTS_HEADER, encoding="utf-8")
    m = result.metrics
    desc = result.candidate.description.replace("\t", " ")
    row = (
        f"{result.experiment_id}\t{result.latency_score:.3f}\t{m.advise_warm_p99_ms:.3f}\t"
        f"{m.closed_loop_ms:.3f}\t{m.closed_loop_reward:.4f}\t{m.confirm_success:.4f}\t"
        f"{m.n_params}\t{result.status}\t{desc}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def format_summary_block(result: BenchResult, *, version: str = "fly_bench_v1") -> str:
    m = result.metrics
    return "\n".join(
        [
            "---",
            f"{version}: 1",
            f"experiment_id: {result.experiment_id}",
            f"latency_score: {result.latency_score:.3f}",
            f"advise_warm_p99: {m.advise_warm_p99_ms:.3f}",
            f"advise_warm_p50: {m.advise_warm_p50_ms:.3f}",
            f"advise_cold: {m.advise_cold_ms:.3f}",
            f"closed_loop_ms: {m.closed_loop_ms:.3f}",
            f"closed_loop_reward: {m.closed_loop_reward:.4f}",
            f"confirm_success: {m.confirm_success:.4f}",
            f"val_success: {m.val_success:.4f}",
            f"ood_success: {m.ood_success:.4f}",
            f"n_params: {m.n_params}",
            f"gates_pass: {str(result.gates_pass).lower()}",
            f"status: {result.status}",
            "---",
        ]
    )
