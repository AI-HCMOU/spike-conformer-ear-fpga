#!/usr/bin/env python3
"""
Evaluation script for SpikeConformer.
Supports both ANN and SNN inference modes.

Usage:
    python scripts/evaluate.py --config config/default.yaml --checkpoint checkpoints/best.pth
    python scripts/evaluate.py --config config/default.yaml --checkpoint checkpoints/best.pth --snn
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.backbone import build_model
from src.data.dataset import build_dataset
from src.evaluation.metrics import compute_all_metrics
from src.conversion.ann_to_snn import ANNtoSNNConverter
from src.utils.config import load_config
from src.utils.seed import set_seed


def evaluate_ann(model, dataloader, device):
    """Evaluate ANN model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model.get_embeddings(images)
            logits = model.head(model.norm(embeddings))
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    probs = torch.cat(all_probs)

    return compute_all_metrics(preds, labels, probs)


def evaluate_snn(model, dataloader, device, cfg):
    """Evaluate SNN model with spike simulation."""
    conversion_cfg = cfg["conversion"]
    converter = ANNtoSNNConverter(
        timesteps=conversion_cfg["timesteps"],
        percentile=conversion_cfg["percentile"],
        attention_alpha=conversion_cfg["attention_alpha"],
    )

    # We need a calibration dataloader (use first 500 samples)
    calib_dataset = torch.utils.data.Subset(dataloader.dataset, range(min(500, len(dataloader.dataset))))
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)

    print("Converting ANN -> SNN...")
    snn_model = converter.convert(model, calib_loader, device)
    snn_model.eval()

    print(f"Running SNN inference (T={conversion_cfg['timesteps']} timesteps)...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = snn_model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # Energy estimate
    energy = snn_model.get_energy_estimate()
    print(f"\nEnergy estimate: {energy['energy_mj']:.3f} mJ/inference")
    print(f"Total ops: {energy['total_ops']:.2e}")
    print(f"Avg firing rate: {energy['avg_firing_rate']:.3f}")

    return {
        "accuracy": (preds == labels).float().mean().item(),
        "total_samples": len(labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpikeConformer")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--snn", action="store_true", help="Evaluate in SNN mode")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Build test dataset
    _, _, test_dataset = build_dataset(cfg["data"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    if args.snn:
        print("\n=== SNN Evaluation ===")
        results = evaluate_snn(model, test_loader, device, cfg)
    else:
        print("\n=== ANN Evaluation ===")
        results = evaluate_ann(model, test_loader, device)

    print("\n=== Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
