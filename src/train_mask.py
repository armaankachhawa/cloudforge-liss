"""Train the three-class clear/cloud/cloud-shadow segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from src.dataset import MaskPatchDataset
from src.models import CloudMaskUNet
from src.train import resolve_device, seed_everything


def train_mask_model(
    patch_dir: Path,
    checkpoint: Path,
    *,
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    device_name: str = "auto",
    seed: int = 42,
) -> Path:
    seed_everything(seed)
    paths = sorted(patch_dir.glob("*.npz"))
    if len(paths) < 10:
        raise ValueError(
            "At least 10 manually corrected NPZ patches are required for a smoke training run"
        )
    dataset = MaskPatchDataset(paths, augment=True)
    valid_count = max(1, round(len(dataset) * 0.2))
    train_count = len(dataset) - valid_count
    train_set, valid_set = random_split(
        dataset, [train_count, valid_count], generator=torch.Generator().manual_seed(seed)
    )
    device = resolve_device(device_name)
    model = CloudMaskUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    class_weights = torch.tensor([0.20, 0.45, 0.35], device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    best = float("inf")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels, weight=class_weights, ignore_index=255)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
        model.eval()
        valid_loss = 0.0
        valid_batches = 0
        with torch.inference_mode():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                valid_loss += float(
                    F.cross_entropy(model(images), labels, weight=class_weights, ignore_index=255)
                )
                valid_batches += 1
        valid_loss /= max(valid_batches, 1)
        train_average = train_loss / max(len(train_loader), 1)
        print(f"epoch={epoch:03d} train={train_average:.5f} valid={valid_loss:.5f}")
        if valid_loss < best:
            best = valid_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": ["clear", "cloud", "cloud_shadow"],
                    "epoch": epoch,
                    "validation_loss": best,
                },
                checkpoint,
            )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CloudForge-LISS mask U-Net")
    parser.add_argument(
        "--patch-dir", type=Path, default=Path("data/processed/masks/manual_patches")
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cloud_mask/best.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        train_mask_model(
            args.patch_dir,
            args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device_name=args.device,
        )
    )


if __name__ == "__main__":
    main()
