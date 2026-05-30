#!/usr/bin/env python3
"""
Training script for SpikeConformer (Phase 1: Conformer backbone training).

Usage:
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --config config/default.yaml --resume checkpoints/last.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.backbone import build_model
from src.models.losses import ArcFaceLoss
from src.data.dataset import build_dataset
from src.training.trainer import Trainer
from src.training.scheduler import CosineWithWarmup
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train SpikeConformer Conformer backbone")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42), cfg.get("deterministic", True))

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = build_model(cfg["model"]).to(device)
    params = model.get_param_count()
    print(f"Model: {cfg['model']['name']}")
    print(f"Parameters: {params['total'] / 1e6:.2f}M (trainable: {params['trainable'] / 1e6:.2f}M)")

    # Build dataset
    train_dataset, val_dataset, test_dataset = build_dataset(cfg["data"])
    train_cfg = cfg["training"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler: Cosine with warmup
    scheduler = CosineWithWarmup(
        optimizer,
        warmup_epochs=train_cfg["warmup_epochs"],
        total_epochs=train_cfg["epochs"],
    )

    # Loss: ArcFace
    loss_params = train_cfg.get("loss_params", {})
    criterion = ArcFaceLoss(
        num_classes=cfg["model"]["num_classes"],
        embed_dim=cfg["model"]["embed_dim"],
        scale=loss_params.get("scale", 30.0),
        margin=loss_params.get("margin", 0.5),
    ).to(device)

    # TensorBoard
    writer = SummaryWriter("runs/conformer_training")

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        cfg=cfg,
        writer=writer,
    )

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume(args.resume)

    # Train
    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    trainer.fit(train_loader, val_loader, start_epoch=start_epoch)

    # Final test evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    print("\n=== Final Test Results ===")
    results = trainer.evaluate(test_loader)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    writer.close()
    print("\nTraining complete. Best checkpoint: checkpoints/best.pth")


if __name__ == "__main__":
    main()
