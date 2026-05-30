"""
SNN Model wrapper for inference with LIF neurons over T timesteps.
"""

import torch
import torch.nn as nn
from typing import Dict
from .lif_neuron import LIFNeuron, PoissonEncoder, SpikeCountDecoder


class SpikingConformer(nn.Module):
    """
    Spiking Neural Network wrapper around the trained Conformer.

    Performs inference by:
    1. Encoding input as Poisson spike trains
    2. Processing through LIF-neuron-equipped layers for T timesteps
    3. Decoding output via spike counting

    Note: This is a simulation-level implementation. For FPGA deployment,
    the hardware RTL (hardware/rtl/) implements the equivalent logic.
    """

    def __init__(
        self,
        ann_model: nn.Module,
        thresholds: Dict[str, float],
        timesteps: int = 48,
        leak_factor: float = 0.95,
    ):
        super().__init__()
        self.ann_model = ann_model
        self.thresholds = thresholds
        self.timesteps = timesteps
        self.leak_factor = leak_factor

        # Input encoder
        self.encoder = PoissonEncoder()

        # Output decoder
        num_classes = ann_model.num_classes
        self.decoder = SpikeCountDecoder(num_classes)

        # Create LIF neurons for each layer
        self.lif_layers: Dict[str, LIFNeuron] = {}
        for name, threshold in thresholds.items():
            self.lif_layers[name] = LIFNeuron(
                threshold=threshold,
                leak=leak_factor,
            )

        # Track firing rates for energy estimation
        self.total_spikes = 0
        self.total_neurons = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SNN inference over T timesteps.

        Args:
            x: Input image (B, C, H, W), normalized to [0, 1]

        Returns:
            Spike rate logits (B, num_classes) - can be used with argmax or softmax
        """
        self._reset_states()
        B = x.shape[0]

        # Normalize input to [0, 1] for Poisson encoding
        # Undo ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_unnorm = x * std + mean
        x_unnorm = x_unnorm.clamp(0.0, 1.0)

        # Simulate over T timesteps
        for t in range(self.timesteps):
            # Encode input as spikes for this timestep
            input_spikes = self.encoder(x_unnorm)

            # Forward through spiking network
            # Use the ANN weights but replace activations with LIF dynamics
            output_spikes = self._spiking_forward(input_spikes)

            # Accumulate output spikes
            self.decoder.accumulate(output_spikes)

        # Return firing rates as logits
        return self.decoder.get_rates(self.timesteps)

    def _spiking_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single-timestep forward pass through spiking network.

        This simplified implementation routes spikes through the ANN's
        linear transformations (weights) and applies LIF dynamics at
        each activation point.
        """
        # For simulation purposes, we use a rate-coding approximation:
        # Feed spikes through ANN layers, applying LIF at activation boundaries
        model = self.ann_model

        # Patch embedding (linear projection - no activation to spike-ify)
        tokens = model.patch_embed(x)
        cnn_feat = x

        lif_idx = 0
        lif_keys = list(self.lif_layers.keys())

        for i, (cnn_stage, trans_stage, fusion, cnn_proj) in enumerate(zip(
            model.cnn_stages, model.transformer_stages,
            model.fusion_modules, model.cnn_projections
        )):
            # CNN stage - apply LIF after each sub-operation
            cnn_feat = self._apply_with_lif(cnn_stage, cnn_feat, lif_keys, lif_idx)
            lif_idx += 1

            # Transformer stage
            for layer in trans_stage:
                tokens = self._apply_with_lif(layer, tokens, lif_keys, lif_idx)
                lif_idx += 1

            # Fusion
            cnn_proj_feat = cnn_proj(cnn_feat)
            tokens = fusion(cnn_proj_feat, tokens)

        # Classification head
        cls_token = tokens.mean(dim=1)
        cls_token = model.norm(cls_token)
        output = model.head(cls_token)

        # Final LIF layer for output spikes
        if lif_idx < len(lif_keys):
            lif = self.lif_layers[lif_keys[lif_idx]]
            output = lif(output)

        return output

    def _apply_with_lif(self, module: nn.Module, x: torch.Tensor,
                        lif_keys: list, idx: int) -> torch.Tensor:
        """Apply a module and then LIF neuron activation."""
        x = module(x)
        if idx < len(lif_keys):
            lif = self.lif_layers[lif_keys[idx]]
            x = lif(x)
        return x

    def _reset_states(self):
        """Reset all neuron membrane potentials."""
        for lif in self.lif_layers.values():
            lif.reset_state()
        self.decoder.reset()
        self.total_spikes = 0
        self.total_neurons = 0

    def get_energy_estimate(self) -> Dict[str, float]:
        """
        Estimate energy consumption based on spike statistics.

        Returns dict with:
            - total_ops: Total accumulate operations
            - energy_mj: Estimated energy in millijoules
            - avg_firing_rate: Average neuron firing rate
        """
        # Based on paper: 270M accumulate ops, 3.1 pJ per op
        e_ac = 3.1e-12  # Joules per accumulate op
        total_synapses = sum(p.numel() for p in self.ann_model.parameters()) // 2
        avg_rate = 0.12  # Empirical from paper
        total_ops = total_synapses * avg_rate * self.timesteps
        energy_j = total_ops * e_ac
        energy_mj = energy_j * 1000

        return {
            "total_ops": total_ops,
            "energy_mj": energy_mj,
            "avg_firing_rate": avg_rate,
            "timesteps": self.timesteps,
        }
