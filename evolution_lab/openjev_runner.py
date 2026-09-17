"""OpenJev Route A backend for evolution-lab genomes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .schema import ExperimentGenome, GenomeError


def openjev_available() -> bool:
    try:
        import openjev_phase1.jevlike.model  # noqa: F401

        return True
    except ImportError:
        return False


def _require_openjev() -> None:
    if not openjev_available():
        raise GenomeError(
            "openjev backend requires openjev-phase1; install with "
            "`pip install -e '.[openjev]'` (see pyproject optional-deps)"
        )


def genome_to_config(genome: ExperimentGenome) -> dict[str, Any]:
    family = genome.architecture.family
    tr = genome.training
    if family == "jev_tiny":
        encoder = "tiny"
    elif family == "jev_hf_head":
        encoder = "hf"
    else:
        raise GenomeError(f"family {family} is not an OpenJev Route A scorer")
    return {
        "encoder": encoder,
        "hf_model": tr.jev_hf_model,
        "width": int(genome.architecture.hidden),
        "rank": int(tr.jev_rank),
        "context_tokens": int(tr.jev_context_tokens),
        "option_tokens": int(tr.jev_option_tokens),
    }


def count_trainable_params(state_dict: dict) -> int:
    return int(sum(tensor.numel() for tensor in state_dict.values()))


def train_and_evaluate(
    genome: ExperimentGenome,
    *,
    n_train: int,
    n_val: int,
    n_confirm: int,
    n_ood: int,
    work_dir: Path,
    splits_dir: Path | None = None,
) -> dict[str, Any]:
    """Train a Jev head on locked splits; return evolution-lab metrics."""
    _require_openjev()
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from openjev_phase1.jevlike.data import JsonlDataset
    from openjev_phase1.jevlike.eval import metrics as jev_metrics
    from openjev_phase1.jevlike.model import make_system, select_device, trainable_state
    from openjev_phase1.jevlike.train import mean_loss, move

    from .jev_splits import resolve_jev_splits

    t0 = time.perf_counter()
    paths = resolve_jev_splits(
        genome,
        n_train=n_train,
        n_val=n_val,
        n_confirm=n_confirm,
        n_ood=n_ood,
        work_dir=work_dir,
        splits_dir=splits_dir,
    )
    config = genome_to_config(genome)
    device = select_device(genome.training.jev_device)
    torch.manual_seed(genome.training.seed)
    model, collator = make_system(config, device)

    train_loader = DataLoader(
        JsonlDataset(paths.train),
        batch_size=genome.training.jev_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        JsonlDataset(paths.val),
        batch_size=genome.training.jev_batch_size,
        collate_fn=collator,
    )
    confirm_loader = DataLoader(
        JsonlDataset(paths.confirm),
        batch_size=genome.training.jev_batch_size,
        collate_fn=collator,
    )
    ood_loader = DataLoader(
        JsonlDataset(paths.ood),
        batch_size=genome.training.jev_batch_size,
        collate_fn=collator,
    )

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=genome.training.jev_lr, weight_decay=1e-4)
    best_state = trainable_state(model)
    best_val = float("inf")
    for _epoch in range(genome.training.jev_epochs):
        model.train()
        for host_batch in train_loader:
            batch = move(host_batch, device)
            loss = F.cross_entropy(model(batch), batch["labels"])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
        validation_loss = mean_loss(model, val_loader, device)
        if validation_loss < best_val:
            best_val = validation_loss
            best_state = trainable_state(model)

    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ckpt_dir / f"{genome.id}.pt"
    torch.save({"config": config, "state_dict": best_state}, checkpoint)

    model.load_state_dict(best_state, strict=False)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    confirm = jev_metrics(model, confirm_loader, device)
    ood = jev_metrics(model, ood_loader, device, shuffle_context=True)
    val = jev_metrics(model, val_loader, device)
    latency = time.perf_counter() - t1
    n_params = count_trainable_params(best_state)
    success = float(confirm["top1"])
    ood_score = float(ood["top1"])
    val_success = float(val["top1"])
    cost = _cost(n_params, fit_s + latency)
    return {
        "success_rate": success,
        "val_success": val_success,
        "ood_score": ood_score,
        "params": n_params,
        "latency_s": latency,
        "fit_s": fit_s,
        "cost": cost,
        "violations": 0.0,
        "joules_per_success": None,
        "joules_unknown": True,
        "jev_checkpoint": str(checkpoint),
        "jev_confirm_top3": float(confirm["top3"]),
        "jev_ood_shuffle_top1": float(ood["top1"]),
        "jev_val_ece": float(val["ece"]),
        "device": str(device),
    }


def _cost(n_params: int, latency_s: float) -> float:
    import numpy as np

    param_term = np.tanh(n_params / 50_000.0)
    time_term = np.tanh(latency_s / 2.0)
    return float(0.7 * param_term + 0.3 * time_term)
