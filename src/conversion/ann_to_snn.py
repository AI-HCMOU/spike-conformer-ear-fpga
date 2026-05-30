"""
ANN-to-SNN Conversion with Attention-Aware Threshold Calibration.

Converts the trained Conformer to a spiking equivalent by:
1. Layer-wise threshold calibration (99.9th percentile)
2. Attention-aware scaling (alpha=0.7) for MHSA layers
3. Replacing ReLU/GELU with LIF neurons
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from pathlib import Path

from .lif_neuron import LIFNeuron
from .snn_model import SpikingConformer


def convert_ann_to_snn(
    model: nn.Module,
    calibration_loader,
    device: torch.device,
    timesteps: int = 48,
    leak_factor: float = 0.95,
    attention_alpha: float = 0.7,
    percentile: float = 99.9,
    num_samples: int = 1000,
) -> SpikingConformer:
    """
    Convert trained Conformer ANN to SNN.

    Args:
        model: Trained Conformer model
        calibration_loader: DataLoader for threshold calibration
        device: Computation device
        timesteps: Number of simulation timesteps T
        leak_factor: LIF neuron leak (beta)
        attention_alpha: Scaling factor for attention layer thresholds
        percentile: Percentile for threshold calibration
        num_samples: Number of calibration samples

    Returns:
        SpikingConformer model ready for inference
    """
    model.eval()
    model.to(device)

    # Step 1: Collect activation statistics
    print("Step 1/3: Collecting activation statistics...")
    activation_stats = _collect_activations(model, calibration_loader, device, num_samples)

    # Step 2: Compute thresholds
    print("Step 2/3: Computing layer-wise thresholds...")
    thresholds = _compute_thresholds(
        activation_stats, percentile, attention_alpha
    )

    # Step 3: Build spiking model
    print("Step 3/3: Building spiking model...")
    snn = SpikingConformer(
        ann_model=model,
        thresholds=thresholds,
        timesteps=timesteps,
        leak_factor=leak_factor,
    )

    print(f"Conversion complete. Timesteps: {timesteps}, Layers: {len(thresholds)}")
    return snn


def _collect_activations(
    model: nn.Module,
    loader,
    device: torch.device,
    num_samples: int,
) -> Dict[str, list]:
    """Hook into model layers and collect activation magnitudes."""
    activations: Dict[str, list] = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.setdefault(name, []).append(
                    output.detach().abs().cpu()
                )
        return hook_fn

    # Register hooks on all activation-producing layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.GELU, nn.ReLU, nn.SiLU)):
            hooks.append(module.register_forward_hook(make_hook(name)))
        elif "softmax" in name.lower() or "attn" in name.lower():
            if isinstance(module, nn.Linear) and "proj" in name:
                hooks.append(module.register_forward_hook(make_hook(f"attn_{name}")))

    # Forward pass on calibration data
    collected = 0
    with torch.no_grad():
        for inputs, _ in loader:
            if collected >= num_samples:
                break
            inputs = inputs.to(device)
            model(inputs)
            collected += inputs.shape[0]

    # Remove hooks
    for h in hooks:
        h.remove()

    return activations


def _compute_thresholds(
    activation_stats: Dict[str, list],
    percentile: float = 99.9,
    attention_alpha: float = 0.7,
) -> Dict[str, float]:
    """
    Compute firing thresholds per layer.

    Standard layers: V_th = Percentile_99.9(|activations|)
    Attention layers: V_th = alpha * Percentile_99.9(|activations|)
    """
    thresholds = {}

    for name, act_list in activation_stats.items():
        all_acts = torch.cat([a.flatten() for a in act_list]).numpy()
        threshold = float(np.percentile(all_acts, percentile))

        # Apply attention-aware scaling
        if "attn" in name.lower():
            threshold *= attention_alpha

        # Ensure non-zero threshold
        threshold = max(threshold, 1e-5)
        thresholds[name] = threshold

    return thresholds


def save_snn(snn: SpikingConformer, path: str):
    """Save converted SNN model."""
    state = {
        "thresholds": snn.thresholds,
        "timesteps": snn.timesteps,
        "leak_factor": snn.leak_factor,
        "ann_state_dict": snn.ann_model.state_dict(),
    }
    torch.save(state, path)
    print(f"SNN saved to {path}")


def load_snn(path: str, model_builder, device: torch.device) -> SpikingConformer:
    """Load saved SNN model."""
    state = torch.load(path, map_location=device, weights_only=False)
    ann_model = model_builder()
    ann_model.load_state_dict(state["ann_state_dict"])
    snn = SpikingConformer(
        ann_model=ann_model,
        thresholds=state["thresholds"],
        timesteps=state["timesteps"],
        leak_factor=state["leak_factor"],
    )
    return snn
