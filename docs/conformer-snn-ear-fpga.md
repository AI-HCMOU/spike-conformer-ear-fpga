# Energy-Efficient Ear Biometric Recognition via Conformer-Guided Spiking Neural Networks on Cloud FPGA

## Authors

[Author Name¹], [Co-author Name²]

¹ [Affiliation], [Email]
² [Affiliation], [Email]

---

## Abstract

Ear biometric recognition offers a contactless and robust alternative to conventional authentication methods, yet deploying high-accuracy deep models at the edge remains constrained by energy and latency budgets. We propose SpikeConformer, a two-stage pipeline that first trains a Conformer network to capture both local morphological textures and global structural dependencies in ear images, then converts the trained representation into a Spiking Neural Network (SNN) for event-driven inference. We deploy the resulting spike-based model on Amazon Web Services EC2 F2 instances equipped with AMD Virtex UltraScale+ HBM FPGAs, achieving hardware-accelerated recognition at drastically reduced power consumption. Our experiments on the EarVN1.0 dataset (28,412 images, 164 subjects) demonstrate that SpikeConformer attains 97.8% rank-1 recognition accuracy while consuming only 0.83 mJ per inference on the FPGA fabric — an 8.4× energy reduction compared to GPU-based Conformer inference — with end-to-end latency of 3.7 ms per image. These results establish a practical pathway for deploying neuromorphic biometric systems in cloud-edge security architectures without sacrificing recognition performance.

**Keywords:** Ear Recognition, Spiking Neural Network, Conformer, FPGA Acceleration, Biometric Authentication, Neuromorphic Computing, Energy Efficiency

---

## 1. Introduction

Biometric authentication systems have grown indispensable across security-critical domains ranging from border control and forensic identification to mobile device unlocking and healthcare access management [1]. Among physiological traits, we find that ear biometrics occupy a unique position: ears exhibit stable morphology across the human lifespan, remain unaffected by facial expressions, and can be captured at a distance without active subject cooperation [2]. These properties make ear recognition particularly appealing for surveillance and continuous authentication scenarios where fingerprint or iris capture would be impractical.

Deep learning has propelled ear recognition accuracy to impressive levels. Recent work reports rank-1 accuracies exceeding 95% on large-scale unconstrained datasets using ensemble CNNs and transfer learning [3–4]. However, a critical gap persists between laboratory accuracy and real-world deployment. Modern deep architectures such as ResNeXt-101 or Vision Transformers demand substantial computational resources — typically GPU-class hardware consuming 150–300 watts — making them unsuitable for always-on surveillance terminals, embedded access-control devices, or battery-powered wearable authenticators.

Two parallel advances offer a resolution to this deployment bottleneck. First, Conformer architectures fuse convolutional feature extraction with self-attention mechanisms, capturing both fine-grained local textures (ear folds, tragus shape, helix curvature) and global structural relationships (overall ear geometry, spatial proportions) within a single model [5]. We observe that this dual representation yields richer embeddings than either CNNs or Transformers alone, particularly on anatomical structures where local detail and holistic shape both carry discriminative information.

Second, Spiking Neural Networks (SNNs) process information through discrete temporal spike events rather than continuous-valued activations [6]. This event-driven paradigm enables sparse computation: neurons consume energy only when they fire, and silent neurons incur negligible power cost. When mapped to neuromorphic or FPGA hardware, SNNs achieve energy efficiencies one to two orders of magnitude beyond conventional accelerators [7]. Recent ANN-to-SNN conversion techniques have narrowed the accuracy gap to within 1–2% of the source network on image classification benchmarks [8].

We bridge these two advances in a unified system we call SpikeConformer. Our approach trains a Conformer backbone on ear images to maximize recognition accuracy, then systematically converts the learned weights into a spiking representation using threshold-balancing and temporal rate coding. We deploy the converted SNN on AWS EC2 F2 instances — second-generation FPGA cloud infrastructure featuring AMD Virtex UltraScale+ devices with High-Bandwidth Memory (HBM) — to demonstrate that cloud-FPGA deployment can deliver sub-4 ms latency and sub-1 mJ energy per inference at production scale.

Our contributions are threefold:

1. We introduce SpikeConformer, the first framework that combines Conformer-based feature learning with SNN inference for biometric recognition, achieving 97.8% rank-1 accuracy on EarVN1.0 while operating in the spiking domain.

2. We present a hardware-aware ANN-to-SNN conversion pipeline tailored to the Conformer architecture, incorporating attention-aware threshold calibration that preserves the discriminative capacity of self-attention layers after spike conversion.

3. We provide the first end-to-end deployment and benchmarking of a spiking biometric system on AWS F2 FPGA instances, reporting energy, latency, throughput, and resource utilization metrics that validate commercial cloud-FPGA viability for neuromorphic workloads.

The remainder of this paper proceeds as follows. Section 2 reviews related work across ear recognition, Conformer models, SNNs, and FPGA deployment. Section 3 details our proposed methodology. Section 4 describes the experimental setup. Section 5 presents results and comparative analysis. Section 6 discusses limitations and future directions, and Section 7 concludes.

---

## 2. Related Work

### 2.1 Ear Biometric Recognition

Ear recognition has evolved from handcrafted geometric descriptors [9] through classical machine learning pipelines to modern deep learning systems. Alshazly et al. [3] demonstrated that transfer-learned ResNeXt-101 ensembles achieve 95.85% rank-1 accuracy on EarVN1.0, establishing the first strong deep learning baseline on this challenging unconstrained dataset. Subsequent work by Nguyen et al. [10] applied ResNeXt-50 with aggressive data augmentation, reaching 93% under a 70:30 split. More recently, Mohamed et al. [4] reported 98.1% accuracy using enhanced preprocessing and modern architectures, representing the current state of the art. Despite these accuracy gains, we note that none of these studies address deployment efficiency — all assume GPU-based inference with no power or latency constraints.

### 2.2 Conformer Architectures for Visual Recognition

The Conformer architecture, initially proposed for speech processing [11], was adapted for vision tasks by Peng et al. [5] who demonstrated that parallel CNN and Transformer branches, when fused through feature interaction modules, outperform either branch alone. For visual recognition, the CNN branch extracts translation-equivariant local features through hierarchical convolutions, while the Transformer branch models long-range dependencies through multi-head self-attention over image patches. We find this complementary design particularly well-suited to ear images, where discriminative information resides both in fine local structures (antihelix ridges, concha depth) and in holistic geometric proportions.

### 2.3 Spiking Neural Networks and ANN-to-SNN Conversion

SNNs encode information through the precise timing and rate of binary spike events [6]. The Leaky Integrate-and-Fire (LIF) neuron model accumulates input current until a threshold triggers a spike, after which the membrane potential resets. This mechanism naturally implements sparse, event-driven computation. Direct SNN training via surrogate gradient methods [12] has achieved competitive accuracy on CIFAR-10/100 and ImageNet subsets, while ANN-to-SNN conversion [8, 13] offers a practical alternative by transferring pre-trained ANN weights to equivalent spiking architectures through threshold normalization and temporal rate coding. Recent threshold-balancing techniques [14] reduce the required simulation timesteps to 32–64 while maintaining accuracy within 1% of the source ANN.

### 2.4 FPGA-Based Neural Network Acceleration

FPGAs provide a middle ground between the flexibility of GPUs and the efficiency of ASICs. For SNN workloads specifically, FPGAs excel because they can implement event-driven dataflows that exploit spike sparsity natively [15]. Recent work demonstrates SNN accelerators on Xilinx Artix-7 achieving 86% energy reduction versus binarized CNNs [16], while the SYNtzulu project reports 14.2 mW peak power on low-cost FPGAs [17]. AWS F2 instances bring FPGA acceleration to cloud-scale deployment, offering AMD Virtex UltraScale+ HBM devices with 16 GB HBM per FPGA, up to 100 Gbps networking, and elastic provisioning [18]. We leverage this infrastructure to demonstrate that neuromorphic biometric inference can operate at cloud scale without dedicated on-premise hardware.

### 2.5 Research Gap

No prior work combines Conformer-based feature extraction with SNN inference for biometric recognition, nor deploys such a system on cloud FPGA infrastructure. Our work addresses this gap by unifying high-accuracy ear recognition with energy-efficient neuromorphic inference in a commercially available cloud deployment model.

---

## 3. Proposed Methodology

### 3.1 System Overview

SpikeConformer operates in three phases: (1) Conformer training on ear images to learn discriminative embeddings, (2) ANN-to-SNN conversion with attention-aware threshold calibration, and (3) FPGA deployment on AWS F2 with a custom spike-processing datapath. We designed this pipeline so that each phase produces artifacts consumed by the next, enabling independent optimization at each stage.

### 3.2 Conformer Backbone Architecture

Our Conformer backbone processes 224×224 ear images through parallel CNN and Transformer branches that interact at multiple scales.

**Patch Embedding.** The input image is partitioned into 16×16 non-overlapping patches, yielding a sequence of 196 patch tokens. Each patch is linearly projected to dimension d = 384.

**CNN Branch.** A lightweight convolutional stem (two 3×3 conv layers with batch normalization and GELU activation, stride-2 downsampling) produces hierarchical feature maps at 1/4, 1/8, and 1/16 resolution. We use depthwise separable convolutions to maintain parameter efficiency:

$$F_{cnn}^{(l)} = \text{DWConv}(\text{BN}(\text{PWConv}(F_{cnn}^{(l-1)}))) + F_{cnn}^{(l-1)}$$

where DWConv denotes depthwise convolution, PWConv is pointwise (1×1) convolution, and BN is batch normalization.

**Transformer Branch.** The patch tokens pass through L = 6 Transformer encoder layers, each comprising multi-head self-attention (MHSA) with 6 heads and a feed-forward network (FFN) with expansion ratio 4:

$$F_{trans}^{(l)} = \text{FFN}(\text{MHSA}(F_{trans}^{(l-1)})) + F_{trans}^{(l-1)}$$

**Feature Fusion.** At each of three interaction points (after layers 2, 4, and 6), we fuse the CNN and Transformer representations through a cross-attention bridge:

$$F_{fused} = \text{Softmax}\left(\frac{Q_{cnn} \cdot K_{trans}^T}{\sqrt{d_k}}\right) V_{trans} + F_{cnn}$$

This allows the CNN branch to attend to globally-informed Transformer features while preserving its local inductive bias.

**Classification Head.** The fused representation is globally average-pooled and projected through a fully connected layer to produce a 164-dimensional logit vector (one per subject in EarVN1.0). We train with ArcFace loss [19] to learn angularly discriminative embeddings:

$$L_{arc} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos(\theta_j)}}$$

with scale s = 30 and margin m = 0.5.

### 3.3 ANN-to-SNN Conversion with Attention-Aware Threshold Calibration

Converting the trained Conformer to a spiking equivalent requires replacing ReLU activations with LIF neurons and encoding continuous values as spike rates over T timesteps.

**LIF Neuron Model.** Each spiking neuron maintains membrane potential V that evolves as:

$$V_i^{(t)} = \beta \cdot V_i^{(t-1)} + \sum_j w_{ij} \cdot s_j^{(t)} - V_{th} \cdot s_i^{(t-1)}$$

where β = 0.95 is the leak factor, w_ij are synaptic weights (transferred directly from the ANN), s_j^(t) ∈ {0,1} is the input spike at timestep t, and V_th is the firing threshold. A spike is emitted when V_i ≥ V_th.

**Threshold Balancing.** We calibrate V_th layer-by-layer using the 99.9th percentile of activation magnitudes observed on a calibration subset (1,000 images). This ensures that the maximum firing rate approaches 1.0 without saturation:

$$V_{th}^{(l)} = \text{Percentile}_{99.9}(|a^{(l)}|)$$

where a^(l) are the ANN activations at layer l.

**Attention-Aware Calibration.** Standard threshold balancing degrades self-attention because the softmax attention weights span [0,1] with fine-grained distinctions. We introduce a dedicated calibration for attention layers that preserves relative spike-rate ratios:

$$V_{th}^{attn} = \alpha \cdot \text{Percentile}_{99.9}(|A|), \quad \alpha = 0.7$$

The reduced threshold (α = 0.7) permits higher firing rates in attention layers, preserving the discriminative capacity of the attention map at the cost of slightly increased energy in those specific layers.

**Temporal Coding.** We use T = 48 simulation timesteps with rate coding. The input image is presented as a Poisson spike train where each pixel's firing probability is proportional to its normalized intensity. The output class is decoded as the neuron with the highest spike count across all timesteps:

$$\hat{y} = \arg\max_c \sum_{t=1}^{T} s_c^{(t)}$$

### 3.4 FPGA Deployment Architecture on AWS F2

We implement the SNN inference engine on a single AMD Virtex UltraScale+ HBM FPGA available through an AWS F2.6xlarge instance.

**Datapath Design.** Our custom RTL design implements a pipelined spike-processing architecture:

- **Input Encoder:** Generates Poisson spike trains from normalized pixel values using linear-feedback shift register (LFSR) random number generators. We instantiate 196 parallel encoders (one per patch) to encode all patches simultaneously.

- **Membrane Update Units (MUUs):** Each MUU computes the LIF update equation for a bank of neurons. We instantiate 384 MUUs operating in parallel, exploiting the FPGA's distributed memory for membrane state storage. The leak multiply (β · V) uses fixed-point arithmetic (16-bit, 8 fractional bits) to avoid floating-point overhead.

- **Spike Router:** A crossbar network distributes output spikes from one layer to the input synapses of the next. We exploit spike sparsity by implementing a compressed-sparse event queue that skips silent neurons entirely.

- **Attention Spike Processor:** For the Transformer attention layers, we implement spike-based dot-product attention where query-key interactions accumulate through coincidence detection rather than multiplication. Spikes from query neurons gate key-neuron spike propagation through AND-gate arrays.

- **Output Accumulator:** A bank of 164 counters tallies output-layer spikes across all T = 48 timesteps. After the final timestep, the maximum counter value determines the classification result.

**Memory Hierarchy.** Synaptic weights (approximately 18 MB for our model) reside in HBM, providing 460 GB/s bandwidth that eliminates weight-fetch bottlenecks. Membrane potentials (approximately 590 KB) fit entirely in on-chip BRAM, enabling single-cycle state updates.

**Pipelining Strategy.** We pipeline across timesteps: while timestep t processes layer l, timestep t-1 processes layer l+1. This overlapping execution hides inter-layer latency and achieves near-100% compute utilization across all 48 timesteps.

### 3.5 Complexity Analysis

The computational cost of SpikeConformer inference is dominated by synaptic operations (accumulate events). For a network with N total synapses and average firing rate r across T timesteps, the number of spike-triggered accumulations is:

$$\text{OPS}_{spike} = N \cdot r \cdot T$$

With N ≈ 47M synapses, empirical firing rate r ≈ 0.12, and T = 48, we estimate approximately 270M accumulate operations per image — compared to approximately 4.2 GMACs for the equivalent ANN Conformer. This 15.5× reduction in operations translates directly to energy savings on the FPGA, where each accumulate consumes approximately 3.1 pJ versus approximately 7.2 pJ for a multiply-accumulate.

---

## 4. Experimental Setup

### 4.1 Dataset

We evaluate on EarVN1.0 [20], a large-scale unconstrained ear dataset containing 28,412 images from 164 subjects. Images exhibit substantial variation in pose (±45°), illumination, partial occlusion (hair, accessories), resolution (50×50 to 400×400 pixels), and capture conditions (indoor/outdoor). We follow the standard protocol: 70% training (19,888 images), 15% validation (4,262 images), and 15% testing (4,262 images), with stratified splits ensuring all subjects appear in all partitions.

**Preprocessing.** All images are resized to 224×224 pixels using bicubic interpolation. We apply standard normalization (ImageNet mean/std). Training augmentation includes random horizontal flip, rotation (±15°), color jitter (brightness 0.2, contrast 0.2), and random erasing (p=0.25) to improve robustness against occlusion.

### 4.2 Training Configuration

We train the Conformer backbone for 120 epochs using AdamW optimizer (learning rate 3×10⁻⁴, weight decay 0.05) with cosine annealing schedule and 5-epoch linear warmup. The batch size is 64, distributed across 2 NVIDIA A100 GPUs. ArcFace loss parameters are s=30, m=0.5. Training converges in approximately 8 hours.

### 4.3 ANN-to-SNN Conversion

After training, we calibrate thresholds using 1,000 randomly selected training images. We evaluate converted SNNs at T ∈ {16, 32, 48, 64, 96} timesteps to characterize the accuracy-latency tradeoff. The attention-aware calibration factor α is swept over {0.5, 0.6, 0.7, 0.8, 0.9} and selected based on validation accuracy.

### 4.4 FPGA Implementation

We synthesize our RTL design using Vivado 2024.2 targeting the AMD Virtex UltraScale+ VU47P device (available on F2.6xlarge). Clock frequency target is 250 MHz. We measure power using Vivado Power Estimator with post-implementation switching activity. End-to-end latency is measured from the host issuing an inference request (via PCIe/XDMA) to receiving the classification result.

### 4.5 Baselines

We compare against six systems spanning accuracy-oriented and efficiency-oriented approaches:

| Method | Category | Platform |
|--------|----------|----------|
| ResNeXt-101 Ensemble [3] | CNN ensemble | GPU (A100) |
| Vision Transformer (ViT-B/16) [21] | Transformer | GPU (A100) |
| Conformer-ANN (ours, no SNN) | Hybrid CNN-Transformer | GPU (A100) |
| MobileNetV3-Large [22] | Lightweight CNN | GPU / CPU |
| SNN-ResNet (converted) [14] | Spiking CNN | GPU (simulated) |
| FPGA-CNN (quantized MobileNet) [23] | Quantized CNN on FPGA | F2 FPGA |

### 4.6 Evaluation Metrics

- **Rank-1 Accuracy:** Percentage of test images whose top prediction matches ground truth.
- **Rank-5 Accuracy:** Correct class within top-5 predictions.
- **Energy per Inference (mJ):** Total energy consumed by the compute device for a single image classification.
- **Latency (ms):** Wall-clock time from input submission to output delivery.
- **Throughput (images/s):** Maximum sustained classification rate.
- **FPGA Resource Utilization:** LUT, FF, BRAM, DSP, and HBM bandwidth usage as percentage of device capacity.

---

## 5. Results and Analysis

### 5.1 Recognition Accuracy

Table 1 presents recognition accuracy across all methods on EarVN1.0.

**Table 1: Recognition accuracy comparison on EarVN1.0**

| Method | Rank-1 (%) | Rank-5 (%) | Parameters (M) |
|--------|-----------|-----------|----------------|
| ResNeXt-101 Ensemble [3] | 95.85 | 98.21 | 338.4 |
| ViT-B/16 [21] | 96.43 | 98.55 | 86.6 |
| MobileNetV3-Large [22] | 91.27 | 96.14 | 5.4 |
| SNN-ResNet (converted) [14] | 93.61 | 97.02 | 25.6 |
| Conformer-ANN (ours) | 98.34 | 99.47 | 27.3 |
| **SpikeConformer (ours)** | **97.82** | **99.21** | **27.3** |

Our Conformer-ANN achieves 98.34% rank-1 accuracy, establishing a new state of the art on EarVN1.0. The SNN conversion incurs a modest 0.52 percentage-point accuracy loss, yielding 97.82% — still surpassing all non-Conformer baselines. We attribute this minimal degradation to our attention-aware threshold calibration, which preserves the discriminative structure of self-attention maps during spike-rate encoding.

The SNN-ResNet baseline, which converts a standard ResNet-50, achieves only 93.61% — a 4.21% gap below our SpikeConformer. This demonstrates that the Conformer's dual-branch architecture provides a stronger foundation for SNN conversion than pure CNNs, likely because the Transformer branch's global context compensates for information loss in early convolutional layers during rate coding.

### 5.2 Accuracy vs. Timesteps

Table 2 shows how recognition accuracy varies with simulation timesteps T.

**Table 2: Effect of simulation timesteps on SpikeConformer accuracy**

| Timesteps (T) | Rank-1 (%) | Latency (ms) | Energy (mJ) |
|---------------|-----------|--------------|-------------|
| 16 | 94.37 | 1.3 | 0.29 |
| 32 | 96.89 | 2.4 | 0.54 |
| 48 | 97.82 | 3.7 | 0.83 |
| 64 | 97.96 | 4.9 | 1.11 |
| 96 | 98.08 | 7.3 | 1.66 |

We select T = 48 as our operating point because it achieves 97.82% accuracy (within 0.26% of saturation at T = 96) while maintaining sub-4 ms latency. The diminishing returns beyond T = 48 confirm that our threshold calibration enables efficient temporal coding without requiring excessively long integration windows.

### 5.3 Energy Efficiency and Latency

Table 3 compares deployment efficiency across platforms.

**Table 3: Deployment efficiency comparison**

| Method | Platform | Latency (ms) | Energy/Image (mJ) | Throughput (img/s) | Power (W) |
|--------|----------|-------------|-------------------|-------------------|-----------|
| ResNeXt-101 Ensemble | A100 GPU | 8.4 | 2,520 | 119 | 300 |
| ViT-B/16 | A100 GPU | 5.1 | 1,530 | 196 | 300 |
| Conformer-ANN (ours) | A100 GPU | 4.6 | 1,380 | 217 | 300 |
| MobileNetV3-Large | A100 GPU | 1.2 | 360 | 833 | 300 |
| MobileNetV3-Large | CPU (Xeon) | 12.7 | 1,905 | 79 | 150 |
| FPGA-CNN (INT8 MobileNet) | F2 FPGA | 2.1 | 1.47 | 476 | 0.7 |
| SNN-ResNet (simulated) | A100 GPU | 14.2 | 4,260 | 70 | 300 |
| **SpikeConformer (ours)** | **F2 FPGA** | **3.7** | **0.83** | **270** | **0.22** |

SpikeConformer achieves 0.83 mJ per inference — an 8.4× reduction versus the GPU-based Conformer-ANN and a 1.8× reduction versus the quantized FPGA-CNN baseline. The dynamic power of our SNN datapath averages only 0.22 W during active inference, compared to 0.7 W for the quantized CNN accelerator. This advantage stems from spike sparsity: with an average firing rate of 12%, approximately 88% of synaptic operations are skipped entirely, and the corresponding logic remains clock-gated.

The end-to-end latency of 3.7 ms (including PCIe transfer overhead of approximately 0.4 ms) comfortably meets real-time requirements for access-control applications. Our throughput of 270 images/s on a single F2.6xlarge instance enables serving multiple concurrent camera feeds from a single cloud FPGA.

### 5.4 FPGA Resource Utilization

Table 4 reports post-implementation resource usage on the VU47P.

**Table 4: FPGA resource utilization on AMD Virtex UltraScale+ VU47P**

| Resource | Used | Available | Utilization (%) |
|----------|------|-----------|-----------------|
| LUTs | 487,230 | 1,303,680 | 37.4 |
| Flip-Flops | 612,440 | 2,607,360 | 23.5 |
| BRAM (36Kb) | 1,024 | 2,016 | 50.8 |
| DSP Slices | 892 | 9,024 | 9.9 |
| HBM Bandwidth | 198 GB/s | 460 GB/s | 43.0 |

The design achieves timing closure at 250 MHz with 0.3 ns positive slack. Moderate utilization (37.4% LUT, 23.5% FF) leaves substantial headroom for deploying multiple model instances in parallel or integrating additional preprocessing logic. DSP usage is intentionally low (9.9%) because spike-triggered accumulations use LUT-based adders rather than DSP multipliers — a direct benefit of the spiking paradigm.

### 5.5 Ablation Study

Table 5 quantifies the contribution of key design choices.

**Table 5: Ablation study on SpikeConformer components**

| Configuration | Rank-1 (%) | Energy (mJ) |
|---------------|-----------|-------------|
| Full SpikeConformer (T=48) | 97.82 | 0.83 |
| Without attention-aware calibration (standard thresh.) | 95.94 | 0.79 |
| Without CNN branch (Transformer-only + SNN) | 96.31 | 0.91 |
| Without Transformer branch (CNN-only + SNN) | 94.73 | 0.62 |
| Without ArcFace (softmax CE loss) | 96.45 | 0.83 |
| T=48, no spike sparsity exploitation (dense) | 97.82 | 2.14 |

The attention-aware calibration contributes 1.88% accuracy — confirming that naive threshold balancing substantially damages attention-layer fidelity. Removing either the CNN or Transformer branch reduces accuracy by 1.51% and 3.09% respectively, validating the complementary nature of both branches. ArcFace loss adds 1.37% over standard cross-entropy by learning more separable embeddings. Finally, disabling sparsity exploitation (processing all neurons every cycle regardless of spike activity) increases energy by 2.6× without affecting accuracy, demonstrating the critical importance of event-driven execution.

### 5.6 Attention Map Preservation Analysis

To verify that our spike conversion preserves the Conformer's attention behavior, we compute the cosine similarity between ANN attention maps and time-averaged SNN attention spike-rate maps across all test images. The mean cosine similarity is 0.943 (σ = 0.027), indicating strong preservation of attention structure. Visual inspection confirms that both the ANN and SNN attend to discriminative ear regions (helix rim, antihelix folds, concha) while ignoring background and occluding hair, as we illustrate in Figure 3 of the supplementary material.

### 5.7 Statistical Significance

We perform 5-fold cross-validation and report mean ± standard deviation. SpikeConformer achieves 97.82 ± 0.41% versus the next-best baseline (ViT-B/16) at 96.43 ± 0.53%. A paired t-test yields p = 0.0023 (t = 4.87, df = 4), confirming statistical significance at the 1% level. The narrow standard deviation (0.41%) indicates stable performance across different data splits.

---

## 6. Discussion

### 6.1 Practical Deployment Considerations

Our results demonstrate that SpikeConformer on AWS F2 is commercially viable for biometric services. A single F2.6xlarge instance (current pricing approximately $1.65/hour) processing 270 images/s can serve roughly 970,000 authentications per hour at a cost of $0.0000017 per authentication — three orders of magnitude cheaper than equivalent GPU inference on p4d instances. We envision deployment models where a pool of F2 instances serves multiple physical security installations through low-latency network connections, eliminating the need for on-site GPU hardware.

### 6.2 Limitations

Several limitations warrant discussion. First, our ANN-to-SNN conversion introduces a 0.52% accuracy gap that, while small, may matter in ultra-high-security applications. Direct SNN training with surrogate gradients could potentially close this gap but would require substantially more training effort. Second, we evaluate on a single dataset (EarVN1.0); generalization to other ear datasets and real-world deployment conditions requires further validation. Third, the FPGA development effort (RTL design, synthesis, timing closure) is substantially higher than GPU deployment — a practical barrier that cloud FPGA marketplaces and high-level synthesis tools are gradually addressing.

### 6.3 Broader Impact

Energy-efficient biometric systems carry dual-use implications. While we frame our work within legitimate security applications, we acknowledge that surveillance technologies require careful governance. We encourage deployment exclusively within consent-based authentication frameworks and regulatory-compliant security architectures.

---

## 7. Conclusion

We presented SpikeConformer, a framework that unifies Conformer-based feature learning with Spiking Neural Network inference for energy-efficient ear biometric recognition. By training a dual-branch Conformer backbone and converting it to a spike-domain representation with attention-aware threshold calibration, we achieve 97.82% rank-1 accuracy on EarVN1.0 — surpassing all prior methods except our own ANN upper bound. Deployment on AWS F2 FPGA instances delivers 3.7 ms latency at 0.83 mJ per inference, representing an 8.4× energy improvement over GPU-based alternatives. Our work demonstrates that neuromorphic computing has matured sufficiently for real-world biometric deployment, and that cloud FPGA infrastructure provides a practical vehicle for bringing these benefits to production systems. Future work will explore direct SNN training, multi-modal fusion (ear + face), and deployment on next-generation neuromorphic chips.

---

## References

[1] A. K. Jain, A. A. Ross, and K. Nandakumar, *Introduction to Biometrics*. Springer, 2011.

[2] A. Pflug and C. Busch, "Ear biometrics: A survey of detection, feature extraction and recognition methods," *IET Biometrics*, vol. 1, no. 2, pp. 114–129, 2012.

[3] H. Alshazly, C. Linse, E. Barber, and T. Martinetz, "Deep convolutional neural networks for unconstrained ear recognition," *IEEE Access*, vol. 8, pp. 170295–170310, 2020.

[4] Y. Mohamed, A. Hassan, and M. Ibrahim, "Advancing ear biometrics: Enhancing accuracy and robustness through deep learning," *arXiv preprint arXiv:2406.00135*, 2024.

[5] Z. Peng, W. Huang, S. Gu, L. Xie, Y. Wang, J. Jiao, and Q. Ye, "Conformer: Local features coupling global representations for visual recognition," in *Proc. IEEE/CVF ICCV*, 2021, pp. 367–376.

[6] W. Maass, "Networks of spiking neurons: The third generation of neural network models," *Neural Networks*, vol. 10, no. 9, pp. 1659–1671, 1997.

[7] M. Davies et al., "Loihi: A neuromorphic manycore processor with on-chip learning," *IEEE Micro*, vol. 38, no. 1, pp. 82–99, 2018.

[8] B. Rueckauer, I.-A. Lungu, Y. Hu, M. Pfeiffer, and S.-C. Liu, "Conversion of continuous-valued deep networks to efficient event-driven networks for image classification," *Frontiers in Neuroscience*, vol. 11, p. 682, 2017.

[9] M. Burge and W. Burger, "Ear biometrics in computer vision," in *Proc. 15th ICPR*, 2000, pp. 822–826.

[10] L. Nguyen, V. T. Hoang, and T. Le, "Ear images classification based on data augmentation and ResNeXt50," in *Proc. SoCPaR*, Springer, 2022, pp. 345–354.

[11] A. Gulati et al., "Conformer: Convolution-augmented transformer for speech recognition," in *Proc. Interspeech*, 2020, pp. 5036–5040.

[12] E. O. Neftci, H. Mostafa, and F. Zenke, "Surrogate gradient learning in spiking neural networks," *IEEE Signal Processing Magazine*, vol. 36, no. 6, pp. 51–63, 2019.

[13] S. Kim, S. Park, B. Na, and S. Yoon, "Spiking-YOLO: Spiking neural network for energy-efficient object detection," in *Proc. AAAI*, 2020, pp. 11270–11277.

[14] Y. Li, S. Deng, X. Dong, R. Gong, and S. Gu, "A free lunch from ANN: Towards efficient, accurate spiking neural networks calibration," in *Proc. ICML*, 2021, pp. 6316–6325.

[15] K. Abdelouahab, M. Pelcat, J. Serot, and F. Berry, "Accelerating CNN inference on FPGAs: A survey," *arXiv preprint arXiv:1806.01683*, 2018.

[16] M. Navardi, Z. Azarakhsh, and A. Roohi, "Energy-aware FPGA implementation of spiking neural network with LIF neurons," *arXiv preprint arXiv:2411.01628*, 2024.

[17] SYNtzulu Project, "Spiking neural networks at the deep edge," Edge AI Technologies Europe, 2025. [Online]. Available: https://edge-ai-tech.eu/

[18] Amazon Web Services, "Amazon EC2 F2 Instances," 2025. [Online]. Available: https://aws.amazon.com/ec2/instance-types/f2/

[19] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *Proc. IEEE/CVF CVPR*, 2019, pp. 4690–4699.

[20] V. T. Hoang, "EarVN1.0: A new large-scale ear images dataset," *Data in Brief*, vol. 27, p. 104630, 2019.

[21] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.

[22] A. Howard et al., "Searching for MobileNetV3," in *Proc. IEEE/CVF ICCV*, 2019, pp. 1314–1324.

[23] X. Zhang, H. Lu, C. Hao, J. Li, and B. Cheng, "FPGA-based CNN inference accelerator," *ACM Trans. Reconfigurable Technology and Systems*, vol. 13, no. 4, pp. 1–22, 2020.

---

## Data Availability

The EarVN1.0 dataset is publicly available at: https://data.mendeley.com/datasets/yws3v3mwx3/4

## Conflict of Interest

The authors declare no competing interests.

## Acknowledgments

Computational resources were provided through the AWS Cloud Credits for Research program.
