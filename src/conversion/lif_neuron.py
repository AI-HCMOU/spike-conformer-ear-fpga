"""
Leaky Integrate-and-Fire (LIF) Neuron implementation.

V_i(t) = beta * V_i(t-1) + sum_j(w_ij * s_j(t)) - V_th * s_i(t-1)
Spike when V_i >= V_th, then reset.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.

    Replaces activation functions (ReLU/GELU) in the ANN.
    Maintains membrane potential state across timesteps.

    Args:
        threshold: Firing threshold V_th
        leak: Leak factor beta (0.95 in paper)
        reset_mode: 'subtract' (soft reset) or 'zero' (hard reset)
    """

    def __init__(self, threshold: float = 1.0, leak: float = 0.95,
                 reset_mode: str = "subtract"):
        super().__init__()
        self.threshold = threshold
        self.leak = leak
        self.reset_mode = reset_mode
        self.membrane: torch.Tensor | None = None

    def reset_state(self):
        """Reset membrane potential (call at start of each inference)."""
        self.membrane = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single timestep forward pass.

        Args:
            x: Input current (weighted spikes from previous layer)
               Shape matches the layer output (B, ...) 

        Returns:
            spikes: Binary spike tensor, same shape as x
        """
        # Initialize membrane if first timestep
        if self.membrane is None:
            self.membrane = torch.zeros_like(x)

        # Leak + integrate
        self.membrane = self.leak * self.membrane + x

        # Fire
        spikes = (self.membrane >= self.threshold).float()

        # Reset
        if self.reset_mode == "subtract":
            self.membrane = self.membrane - spikes * self.threshold
        else:
            self.membrane = self.membrane * (1.0 - spikes)

        return spikes

    @property
    def firing_rate(self) -> float:
        """Compute average firing rate (for energy estimation)."""
        if self.membrane is None:
            return 0.0
        return 0.0  # Tracked externally


class PoissonEncoder(nn.Module):
    """
    Rate-coded Poisson spike encoder for input images.

    Each pixel fires with probability proportional to its normalized intensity.
    P(spike) = (pixel - min) / (max - min)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate one timestep of Poisson spikes from normalized input.

        Args:
            x: Normalized input image (B, C, H, W), values in [0, 1] ideally

        Returns:
            spikes: Binary spike tensor (B, C, H, W)
        """
        # Clamp to valid probability range
        probs = x.clamp(0.0, 1.0)
        # Sample spikes from Bernoulli distribution
        spikes = torch.bernoulli(probs)
        return spikes


class SpikeCountDecoder(nn.Module):
    """
    Decode output class from spike counts over T timesteps.
    y_hat = argmax_c sum_t s_c(t)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.counts: torch.Tensor | None = None

    def reset(self):
        self.counts = None

    def accumulate(self, spikes: torch.Tensor):
        """Add spikes from one timestep. spikes: (B, num_classes)"""
        if self.counts is None:
            self.counts = torch.zeros_like(spikes)
        self.counts += spikes

    def decode(self) -> torch.Tensor:
        """Return predicted class indices. (B,)"""
        if self.counts is None:
            raise RuntimeError("No spikes accumulated")
        return self.counts.argmax(dim=1)

    def get_rates(self, timesteps: int) -> torch.Tensor:
        """Return firing rates (for soft predictions). (B, num_classes)"""
        if self.counts is None:
            raise RuntimeError("No spikes accumulated")
        return self.counts / timesteps
