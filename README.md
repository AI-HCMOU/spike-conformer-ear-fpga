# SpikeConformer

> Energy-Efficient Ear Biometric Recognition via Conformer-Guided Spiking Neural Networks on Cloud FPGA

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download EarVN1.0 dataset
python scripts/download_data.py --output data/earvn1.0

# 3. Train Conformer backbone
python scripts/train.py --config config/default.yaml

# 4. Convert to SNN
python scripts/convert.py --checkpoint checkpoints/best.pth --timesteps 48

# 5. Evaluate SNN
python scripts/evaluate.py --config config/default.yaml --mode snn --timesteps 48

# 6. (Optional) Generate FPGA bitstream
cd hardware && vivado -mode batch -source build.tcl
```

## Architecture

SpikeConformer is a two-stage pipeline:
1. **Conformer Backbone** (CNN + Transformer dual-branch with cross-attention fusion)
2. **ANN-to-SNN Conversion** (attention-aware threshold calibration, LIF neurons, rate coding)

## Expected Results (from paper)

| Metric | Conformer-ANN | SpikeConformer (T=48) |
|--------|---------------|----------------------|
| Rank-1 Accuracy | 98.34% | 97.82% |
| Rank-5 Accuracy | 99.47% | 99.21% |
| F1-Score | 98.21% | 97.64% |
| Energy/Inference | 1,380 mJ (GPU) | 0.83 mJ (FPGA) |
| Latency | 4.6 ms (GPU) | 3.7 ms (FPGA) |

## Project Structure

```
spikeconformer/
├── config/default.yaml          # All hyperparameters
├── src/
│   ├── models/
│   │   ├── backbone.py          # Conformer architecture
│   │   ├── layers.py            # Custom modules (DWSepConv, CrossAttention)
│   │   └── losses.py            # ArcFace loss
│   ├── data/
│   │   ├── dataset.py           # EarVN1.0 dataset class
│   │   └── augmentation.py      # Augmentation pipeline
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   └── scheduler.py         # Cosine + warmup scheduler
│   ├── conversion/
│   │   ├── ann_to_snn.py        # Threshold calibration + conversion
│   │   ├── lif_neuron.py        # LIF neuron implementation
│   │   └── snn_model.py         # SNN inference wrapper
│   ├── evaluation/
│   │   ├── metrics.py           # Rank-1/5, F1, MCC, AUC
│   │   └── visualization.py     # Plots and attention maps
│   └── utils/
│       ├── config.py            # Config loader
│       └── seed.py              # Reproducibility
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   ├── convert.py               # ANN→SNN conversion
│   └── download_data.py         # Dataset downloader
├── hardware/
│   ├── rtl/spike_core.v         # Verilog SNN inference engine
│   ├── constraints.xdc          # FPGA timing/pin constraints
│   └── build.tcl                # Vivado synthesis script
└── tests/
    ├── test_model.py            # Shape + forward pass tests
    └── test_conversion.py       # SNN equivalence tests
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA 12.x (for GPU training)
- Vivado 2024.2 (for FPGA synthesis, optional)

## Citation

If you use this implementation, please cite the original paper.
