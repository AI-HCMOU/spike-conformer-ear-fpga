"""
Tests for ANN-to-SNN conversion pipeline.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backbone import build_model
from src.conversion.lif_neuron import LIFNeuron, PoissonEncoder, SpikeCountDecoder
from src.conversion.snn_model import SpikingConformer


@pytest.fixture
def model_cfg():
    return {
        "name": "SpikeConformer",
        "num_classes": 10,
        "embed_dim": 128,
        "num_heads": 4,
        "transformer_depth": 2,
        "cnn_channels": [32, 64, 128],
        "patch_size": 16,
        "img_size": 224,
        "dropout": 0.0,
    }


@pytest.fixture
def model(model_cfg):
    return build_model(model_cfg)


class TestLIFNeuron:
    def test_spike_generation(self):
        lif = LIFNeuron(threshold=1.0, leak=0.9)
        # Strong input should eventually produce spikes
        x = torch.ones(1, 10) * 2.0
        total_spikes = 0
        for _ in range(10):
            spikes = lif(x)
            total_spikes += spikes.sum().item()
        assert total_spikes > 0

    def test_no_spike_below_threshold(self):
        lif = LIFNeuron(threshold=10.0, leak=0.5)
        x = torch.ones(1, 10) * 0.1
        # With high threshold and low leak, few timesteps won't accumulate enough
        spikes = lif(x)
        assert spikes.sum().item() == 0

    def test_reset_state(self):
        lif = LIFNeuron(threshold=1.0, leak=0.95)
        x = torch.ones(1, 10)
        lif(x)  # Accumulate membrane potential
        lif.reset_state()
        assert lif.membrane is None

    def test_output_is_binary(self):
        lif = LIFNeuron(threshold=0.5, leak=0.9)
        x = torch.randn(4, 32)
        for _ in range(5):
            spikes = lif(x)
            unique_vals = spikes.unique()
            # Output should only be 0 or 1
            assert all(v in [0.0, 1.0] for v in unique_vals.tolist())


class TestPoissonEncoder:
    def test_output_shape(self):
        encoder = PoissonEncoder()
        x = torch.rand(2, 3, 32, 32)
        spikes = encoder(x)
        assert spikes.shape == x.shape

    def test_output_binary(self):
        encoder = PoissonEncoder()
        x = torch.rand(2, 3, 32, 32)
        spikes = encoder(x)
        unique_vals = spikes.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_rate_proportional_to_input(self):
        encoder = PoissonEncoder()
        # High input should produce more spikes on average
        high_input = torch.ones(1, 1, 100, 100) * 0.9
        low_input = torch.ones(1, 1, 100, 100) * 0.1
        high_spikes = sum(encoder(high_input).sum().item() for _ in range(100))
        low_spikes = sum(encoder(low_input).sum().item() for _ in range(100))
        assert high_spikes > low_spikes


class TestSpikeCountDecoder:
    def test_accumulate_and_rate(self):
        decoder = SpikeCountDecoder(num_classes=10)
        decoder.reset()
        spike_counts = torch.zeros(2, 10)
        spike_counts[0, 3] = 1.0
        spike_counts[1, 7] = 1.0
        for _ in range(10):
            decoder.accumulate(spike_counts)
        rates = decoder.get_rates(timesteps=10)
        assert rates[0, 3] == 1.0
        assert rates[1, 7] == 1.0
        assert rates[0, 0] == 0.0


class TestSpikingConformer:
    def test_energy_estimate(self, model):
        thresholds = {"layer_0": 1.0, "layer_1": 1.5}
        snn = SpikingConformer(model, thresholds, timesteps=8)
        energy = snn.get_energy_estimate()
        assert "energy_mj" in energy
        assert energy["energy_mj"] > 0
        assert energy["timesteps"] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
