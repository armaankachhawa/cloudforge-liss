"""Train the core mask-guided Attention ResUNet reconstruction model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SyntheticLISSDataset
from src.losses import CloudReconstructionLoss
from src.models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CloudReconstructionLoss,
    device: torch.device,
    *,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {"loss": 0.0}
    batches = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, leave=False, desc="train" if training else "valid"):
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                prediction = model(inputs)
                loss, components = criterion(prediction, target, mask)
            if training:
                assert optimizer is not None
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


def train(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config["training"].get("device", "auto"))
    manifest = Path(config["data"]["manifest"])
    train_data = SyntheticLISSDataset(manifest, "train", augment=True)
    valid_data = SyntheticLISSDataset(manifest, "validation", augment=False)
    workers = int(config["training"].get("workers", 0))
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model_config = config.get("model", {})
    model = build_model(
        model_config.get("name", "attention_resunet"),
        int(config["data"]["input_channels"]),
        int(config["data"]["output_channels"]),
        int(model_config.get("base_channels", 32)),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    loss_config = config["loss"]
    criterion = CloudReconstructionLoss(
        loss_config["masked_l1"],
        loss_config["masked_ssim"],
        loss_config["spectral_angle"],
        loss_config["edge"],
    )
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_dir = Path(config["training"].get("checkpoint_dir", "checkpoints/model1"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = float("inf")
    best_path = checkpoint_dir / "best.pt"
    print(f"Training on {device}; AMP={use_amp}; train={len(train_data)} valid={len(valid_data)}")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=use_amp,
        )
        valid_metrics = run_epoch(model, valid_loader, criterion, device, mixed_precision=use_amp)
        record = {"epoch": epoch, "train": train_metrics, "validation": valid_metrics}
        history.append(record)
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.5f} "
            f"validation={valid_metrics['loss']:.5f}"
        )
        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "validation_loss": best_loss,
                },
                best_path,
            )
        (checkpoint_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CloudForge-LISS Model 1")
    parser.add_argument("--config", type=Path, default=Path("configs/model1.yaml"))
    args = parser.parse_args()
    print(f"Best checkpoint: {train(args.config)}")


if __name__ == "__main__":
    main()
