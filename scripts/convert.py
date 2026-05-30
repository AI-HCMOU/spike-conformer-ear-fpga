#!/usr/bin/env python3
"""
ANN-to-SNN conversion script.

Usage:
    python scripts/convert.py --config config/default.yaml --checkpoint checkpoints/best.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.backbone import build_model
from src.data.dataset import build_dataset
from src.conversion.ann_to_snn import ANNtoSNNConverter
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Convert trained ANN to SNN")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="checkpoints/snn_model.pth")
    parser.add_argument("--calib-samples", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load ANN model
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded ANN checkpoint: {args.checkpoint}")

    # Build calibration dataloader
    train_dataset, _, _ = build_dataset(cfg["data"])
    calib_size = min(args.calib_samples, len(train_dataset))
    calib_dataset = torch.utils.data.Subset(train_dataset, range(calib_size))
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
    print(f"Calibration samples: {calib_size}")

    # Convert
    conversion_cfg = cfg["conversion"]
    converter = ANNtoSNNConverter(
        timesteps=conversion_cfg["timesteps"],
        percentile=conversion_cfg["percentile"],
        attention_alpha=conversion_cfg["attention_alpha"],
    )

    print("\nStarting conversion...")
    print(f"  Timesteps: {conversion_cfg['timesteps']}")
    print(f"  Percentile: {conversion_cfg['percentile']}")
    print(f"  Attention alpha: {conversion_cfg['attention_alpha']}")

    snn_model = converter.convert(model, calib_loader, device)

    # Save converted model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "snn_state": snn_model.state_dict(),
        "thresholds": snn_model.thresholds,
        "timesteps": snn_model.timesteps,
        "config": cfg,
    }, str(output_path))

    print(f"\nSNN model saved to: {output_path}")

    # Print energy estimate
    energy = snn_model.get_energy_estimate()
    print(f"\n=== Energy Estimate ===")
    print(f"  Total Ops: {energy['total_ops']:.2e}")
    print(f"  Energy: {energy['energy_mj']:.3f} mJ/inference")
    print(f"  Avg Firing Rate: {energy['avg_firing_rate']:.3f}")
    print(f"  Timesteps: {energy['timesteps']}")


if __name__ == "__main__":
    main()
