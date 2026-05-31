# SpikeConformer

> **Energy-Efficient Ear Biometric Recognition via Conformer-Guided Spiking Neural Networks on Cloud FPGA**

📄 **[Read the full paper (PDF)](docs/conformer-snn-ear-fpga.pdf)**

---

## Overview

SpikeConformer achieves **97.82% Rank-1 accuracy** on ear recognition while consuming only **0.83 mJ per inference** on AWS F2 FPGA — a **1,663× energy reduction** versus GPU deployment with less than 0.52% accuracy loss.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SpikeConformer Pipeline                           │
├─────────────────────┬──────────────────────┬────────────────────────────┤
│    Phase 1: Train   │  Phase 2: Convert    │    Phase 3: Deploy         │
│                     │                      │                            │
│  ┌───────────────┐  │  ┌────────────────┐  │  ┌──────────────────────┐  │
│  │  EarVN1.0     │  │  │ Threshold      │  │  │  AWS F2 FPGA         │  │
│  │  224×224 RGB  │  │  │ Calibration    │  │  │  (VU47P, 250MHz)     │  │
│  └──────┬────────┘  │  │ (99.9th %-ile) │  │  │                      │  │
│         ▼           │  │ α=0.7 attn     │  │  │  384 Membrane Units  │  │
│  ┌───────────────┐  │  └───────┬────────┘  │  │  Spike Routing       │  │
│  │  Conformer    │  │          ▼           │  │  HBM Weights         │  │
│  │  + ArcFace    │──┼──▶ ReLU → LIF       │──┼──▶ 0.22W, 3.7ms      │  │
│  │  (27.3M)      │  │   neurons (T=48)    │  │                      │  │
│  └───────────────┘  │                      │  └──────────────────────┘  │
└─────────────────────┴──────────────────────┴────────────────────────────┘
```

---

## Conformer Backbone

```
Input (224×224×3)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
┌──────────────────┐          ┌──────────────────────┐
│   CNN Branch     │          │  Transformer Branch   │
│                  │          │                       │
│  DWSepConv 96   │          │  Patch Embed (16×16)  │
│  DWSepConv 192  │◄────────►│  6× MHSA (d=384,h=6) │
│  DWSepConv 384  │  Cross   │  FFN (384→1536→384)   │
│                  │  Attn    │                       │
└────────┬─────────┘  ×3     └──────────┬────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │ Fused 384-dim    │
              │ Embedding        │
              └──────────────────┘
```

**Key design choices:**
- Dual-branch captures both local texture (CNN) and global shape (Transformer)
- Cross-attention fusion at 3 scale levels enables bidirectional feature exchange
- ArcFace angular margin loss learns highly separable identity embeddings
- 27.3M parameters — lightweight enough for FPGA deployment

---

## ANN-to-SNN Conversion

```
Trained ANN                          Spiking Neural Network
───────────                          ──────────────────────

ReLU(W·x) = 0.73        ──────▶     LIF neuron fires 35/48 times
                                     firing rate = 35/48 ≈ 0.73 ✓

┌─────────────────────────────────────────────────────────────┐
│  Attention-Aware Threshold Calibration                       │
│                                                             │
│  Standard layers:  θ = percentile(activations, 99.9%)       │
│  Attention layers: θ = α × percentile(activations, 99.9%)  │
│                    where α = 0.7 (preserves attention maps) │
└─────────────────────────────────────────────────────────────┘
```

**Why α=0.7 for attention layers?** Standard calibration degrades softmax attention patterns. Lower thresholds allow more spikes, better approximating the continuous attention distribution. This single trick recovers +1.88% accuracy.

---

## FPGA Deployment Architecture (AWS F2)

```
┌────────────────────────────────────────────────────────────────┐
│  AWS F2 Instance — AMD Virtex UltraScale+ VU47P               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────┐    ┌───────────────────┐    ┌──────────────┐    │
│  │  Poisson │    │  384× Membrane    │    │  Spike-Rate  │    │
│  │  Encoder │───▶│  Update Units     │───▶│  Decoder     │    │
│  │  (Input) │    │  (Pipelined LIF)  │    │  (Output)    │    │
│  └──────────┘    └─────────┬─────────┘    └──────────────┘    │
│                            │                                   │
│                  ┌─────────▼─────────┐                         │
│                  │  Compressed-Sparse │                         │
│                  │  Spike Router      │                         │
│                  │  (skip 88% ops)   │                         │
│                  └─────────┬─────────┘                         │
│                            │                                   │
│       ┌────────────────────┼───────────────────┐               │
│       │                    │                   │               │
│  ┌────▼────┐    ┌──────────▼────────┐   ┌─────▼──────┐        │
│  │  BRAM   │    │  HBM (16GB)       │   │  AND-Gate  │        │
│  │  States │    │  Synaptic Weights  │   │  Attention │        │
│  │(1024×36K)│    │  (14/32 channels)  │   │  Processing│        │
│  └─────────┘    └───────────────────┘   └────────────┘        │
│                                                                │
│  Clock: 250MHz │ WNS: +0.31ns │ Power: 0.22W (dynamic)        │
└────────────────────────────────────────────────────────────────┘
```

---

## Results

### Recognition Accuracy on EarVN1.0

| Method | Rank-1 (%) | Rank-5 (%) | F1 (%) | Params (M) |
|--------|:----------:|:----------:|:------:|:----------:|
| ResNeXt-101 Ensemble | 95.85 | 98.21 | 95.62 | 338.4 |
| ViT-B/16 | 96.43 | 98.55 | 96.28 | 86.6 |
| MobileNetV3-Large | 91.27 | 96.14 | 90.83 | 5.4 |
| SNN-ResNet | 93.61 | 97.02 | 93.24 | 25.6 |
| Conformer-ANN (ours) | 98.34 | 99.47 | 98.21 | 27.3 |
| **SpikeConformer (ours)** | **97.82** | **99.21** | **97.64** | **27.3** |

### Accuracy vs. Timesteps

| T | Rank-1 (%) | Latency (ms) | Energy (mJ) | Fire Rate |
|:-:|:----------:|:------------:|:-----------:|:---------:|
| 16 | 94.37 | 1.3 | 0.29 | 0.14 |
| 32 | 96.89 | 2.4 | 0.54 | 0.13 |
| **48** | **97.82** | **3.7** | **0.83** | **0.12** |
| 64 | 97.96 | 4.9 | 1.11 | 0.11 |
| 96 | 98.08 | 7.3 | 1.66 | 0.10 |

> T=48 selected as operating point: within 0.26% of saturation while keeping latency under 4ms.

### Deployment Efficiency Comparison

| Method | Platform | Latency (ms) | Energy (mJ) | Throughput (img/s) | Power (W) | Acc. (%) |
|--------|----------|:------------:|:-----------:|:-----------------:|:---------:|:--------:|
| ResNeXt-101 Ensemble | A100 GPU | 8.4 | 2,520 | 119 | 300 | 95.85 |
| ViT-B/16 | A100 GPU | 5.1 | 1,530 | 196 | 300 | 96.43 |
| Conformer-ANN (ours) | A100 GPU | 4.6 | 1,380 | 217 | 300 | 98.34 |
| MobileNetV3-Large | A100 GPU | 1.2 | 360 | 833 | 300 | 91.27 |
| FPGA-CNN (INT8) | F2 FPGA | 2.1 | 1.47 | 476 | 0.70 | 90.63 |
| SNN-ResNet (sim.) | A100 GPU | 14.2 | 4,260 | 70 | 300 | 93.61 |
| **SpikeConformer (ours)** | **F2 FPGA** | **3.7** | **0.83** | **270** | **0.22** | **97.82** |

### FPGA Resource Utilization (VU47P)

| Resource | Used | Available | Utilization (%) |
|----------|-----:|----------:|:---------------:|
| LUTs | 487,230 | 1,303,680 | 37.4 |
| Flip-Flops | 612,440 | 2,607,360 | 23.5 |
| BRAM (36Kb) | 1,024 | 2,016 | 50.8 |
| DSP Slices | 892 | 9,024 | 9.9 |
| HBM Channels | 14 | 32 | 43.8 |
| *Timing* | | *250 MHz, WNS = +0.31 ns* | |

> Low DSP usage (9.9%) because spike-triggered accumulation uses LUT-based adders instead of DSP multipliers.

### Ablation Study

| Configuration | Rank-1 (%) | Energy (mJ) |
|---------------|:----------:|:-----------:|
| Full SpikeConformer (T=48) | 97.82 | 0.83 |
| w/o attention-aware calibration | 95.94 (−1.88) | 0.79 |
| w/o CNN branch | 96.31 (−1.51) | 0.91 |
| w/o Transformer branch | 94.73 (−3.09) | 0.62 |
| w/o ArcFace loss | 96.45 (−1.37) | 0.83 |
| w/o sparsity exploitation | 97.82 | 2.14 (2.6×) |
| w/o cross-attention fusion | 96.87 (−0.95) | 0.81 |

### Multi-FPGA Scaling

| F2 Instance | FPGAs | Throughput (img/s) | Linear Scaling (%) |
|-------------|:-----:|:-----------------:|:-----------------:|
| f2.6xlarge | 1 | 270 | 100 |
| f2.12xlarge | 2 | 531 | 98.3 |
| f2.48xlarge | 8 | 2,089 | 96.7 |

### Statistical Significance (5-fold CV)

| Method | Rank-1 (%) | p-value vs. Ours |
|--------|:----------:|:----------------:|
| ResNeXt-101 Ens. | 95.85 ± 0.67 | < 0.001 |
| ViT-B/16 | 96.43 ± 0.53 | 0.0023 |
| MobileNetV3-L | 91.27 ± 0.81 | < 0.001 |
| SNN-ResNet | 93.61 ± 0.74 | < 0.001 |
| Conformer-ANN | 98.34 ± 0.38 | 0.087 |
| **SpikeConformer** | **97.82 ± 0.41** | — |

### Comparison with Neuromorphic Hardware (Projected)

| Platform | Energy (mJ) | Latency (ms) | Availability |
|----------|:-----------:|:------------:|:------------:|
| Intel Loihi 2 | ~0.12† | ~8.5 | Research only |
| IBM NorthPole | ~0.08† | ~2.1 | Research only |
| SpiNNaker 2 | ~0.45† | ~12.0 | Limited |
| **AWS F2 (ours)** | **0.83** | **3.7** | **Commercial** |

> †Projected estimates based on published per-spike energy figures scaled to our model's 47M synapses and 12% firing rate.

---

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

---

## FPGA Deployment Guide

### End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  OFFLINE (your machine / GPU server)                                  │
│                                                                      │
│  1. Train Conformer (PyTorch)                                        │
│         → best.pth (float32 weights)                                 │
│                                                                      │
│  2. Convert ANN→SNN (Python)                                         │
│         → weights_fixed.bin (Q8.8 fixed-point weights)               │
│         → thresholds.json (per-layer thresholds)                     │
│                                                                      │
│  3. Synthesize RTL (Vivado)                                          │
│         → spike_core.xclbin (FPGA bitstream)                         │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Upload to AWS
┌──────────────────────────────────────────────────────────────────────┐
│  ONLINE (AWS F2 instance)                                            │
│                                                                      │
│  4. Host Program (CPU) ←→ FPGA Kernel (spike_core)                   │
│                                                                      │
│     ┌─────────────┐         PCIe / DMA          ┌──────────────┐    │
│     │  Host CPU   │ ───── weights_fixed.bin ───▶ │  HBM (FPGA)  │    │
│     │             │ ───── input image ─────────▶ │              │    │
│     │             │ ◀──── 384-dim embedding ──── │  spike_core  │    │
│     └─────────────┘                              └──────────────┘    │
│                                                                      │
│  5. Cosine similarity matching → identity result                     │
└──────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step

| Step | Tool | Input | Output |
|------|------|-------|--------|
| Train | PyTorch | Images | `best.pth` |
| Convert | Python | `best.pth` | `weights_fixed.bin` + `thresholds.json` |
| Synthesize | Vivado | `spike_core.v` | `.xclbin` bitstream |
| Deploy | Host program | Image + `.xclbin` | 384-dim embedding |
| Match | NumPy/CPU | Embedding + gallery | Identity |

### Weight Export (Float32 → Q8.8 Fixed-Point)

```python
# Convert trained weights for FPGA consumption
def float_to_q8_8(tensor):
    """Float32 → 16-bit fixed-point (8 integer, 8 fractional bits)"""
    return (tensor * 256).round().clamp(-32768, 32767).short()

# Export layer-by-layer as flat binary
with open('weights_fixed.bin', 'wb') as f:
    for name in layer_order:
        f.write(float_to_q8_8(model.state_dict()[name]).numpy().tobytes())
```

### FPGA Emulation (Before Real Hardware)

```bash
# 1. Behavioral simulation (logic correctness)
cd hardware
iverilog -o sim rtl/spike_core.v testbench/tb_spike_core.v
vvp sim

# 2. Hardware emulation (cycle-accurate, timing)
v++ --platform xilinx_aws-vu47p_shell-v1 \
    --target hw_emu \
    --kernel spike_core \
    -o spike_core_emu.xclbin

export XCL_EMULATION_MODE=hw_emu
./host_app spike_core_emu.xclbin

# 3. Real hardware build
v++ --platform xilinx_aws-vu47p_shell-v1 \
    --target hw \
    --kernel spike_core \
    -o spike_core.xclbin
```

| Mode | Speed | What it tests | Cost |
|------|-------|---------------|------|
| `sw_emu` | Seconds | Functional correctness | Free (local) |
| `hw_emu` | Minutes–hours | Cycle-accurate timing | Free (local) |
| `hw` | Real-time | Actual FPGA | ~$1.65/hr (F2) |

### AWS F2 Deployment

```bash
# Create Amazon FPGA Image (AFI)
aws ec2 create-fpga-image \
    --input-storage-location Bucket=my-bucket,Key=spike_core.xclbin \
    --name "SpikeConformer-v1"

# Load on F2 instance
sudo fpga-load-local-image -S 0 -I agfi-xxxxx

# Run inference
python host_inference.py --image probe.jpg --gallery gallery/
```

### What Happens Inside the FPGA (Per Inference)

```
For each timestep t = 1..48:
    input_spike[t] = Poisson(image_pixel)    # 0 or 1

    For each neuron i:
        if input_spike = 1:
            membrane[i] += weight[i]          # add only (no multiply!)
        if membrane[i] ≥ threshold:
            output_spike = 1                  # fire
            membrane[i] -= threshold          # reset

After T=48 steps:
    embedding[i] = spike_count[i] / 48       # firing rate = feature value
    → cosine_match(embedding, gallery) → identity
```

---

## Project Structure

```
spikeconformer/
├── config/default.yaml          # All hyperparameters
├── src/
│   ├── models/
│   │   ├── backbone.py          # Conformer architecture
│   │   ├── layers.py            # DWSepConv, CrossAttention, MHSA
│   │   └── losses.py            # ArcFace angular margin loss
│   ├── data/
│   │   ├── dataset.py           # EarVN1.0 dataset class
│   │   └── augmentation.py      # Augmentation pipeline
│   ├── training/
│   │   ├── trainer.py           # Training loop with ArcFace
│   │   └── scheduler.py         # Cosine + warmup scheduler
│   ├── conversion/
│   │   ├── ann_to_snn.py        # Threshold calibration + conversion
│   │   ├── lif_neuron.py        # LIF neuron implementation
│   │   └── snn_model.py         # SNN inference wrapper (T timesteps)
│   ├── evaluation/
│   │   └── metrics.py           # Rank-1/5, F1, MCC, AUC
│   └── utils/
│       ├── config.py            # YAML config loader
│       └── seed.py              # Reproducibility
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # ANN & SNN evaluation
│   ├── convert.py               # ANN→SNN conversion
│   └── download_data.py         # EarVN1.0 dataset downloader
├── hardware/
│   ├── rtl/spike_core.v         # Verilog LIF neuron array (AXI-Stream)
│   ├── constraints.xdc          # VU47P timing constraints (250MHz)
│   └── build.tcl                # Vivado synthesis script
├── docs/
│   ├── conformer-snn-ear-fpga.tex  # Full paper (LaTeX)
│   └── references.bib              # 45 verified references
└── tests/
    ├── test_model.py            # Architecture shape tests
    └── test_conversion.py       # SNN equivalence tests
```

---

## Key Highlights

| | GPU (A100) | FPGA (AWS F2) | Improvement |
|---|:---:|:---:|:---:|
| Energy per inference | 1,380 mJ | 0.83 mJ | **1,663×** |
| Power | 300 W | 0.22 W | **1,364×** |
| Latency | 4.6 ms | 3.7 ms | **1.24×** |
| Accuracy | 98.34% | 97.82% | −0.52% |

---

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA 12.x (for GPU training)
- Vivado 2024.2 (for FPGA synthesis, optional)
- AWS F2 instance (for cloud FPGA deployment, optional)

---

## Dataset

**EarVN1.0** — 28,412 ear images from 164 subjects  
Source: [Mendeley Data](https://data.mendeley.com/datasets/yws3v3mwx3/4) (DOI: 10.17632/yws3v3mwx3.4)

---

## Citation

If you use this implementation, please cite:

```bibtex
@article{spikeconformer2025,
  title   = {Energy-Efficient Ear Biometric Recognition via Conformer-Guided
             Spiking Neural Networks on Cloud FPGA},
  year    = {2025}
}
```

---

## License

This project is provided for academic research purposes.
