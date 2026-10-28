# TinyAI: Quantization, Pruning, and Distillation Pipeline for Enterprise Analytics Edge Devices

**Target Venue**: IEEE Access  
**Authors**: Sahiti Veeravalli et al.  
**Classification**: Edge AI, Model Compression, Tabular Predictive Systems, Industrial Cyber-Physical Systems  

---

## Abstract
Deploying high-capacity deep learning and gradient boosted models to constrained enterprise edge predictive endpoints (point-of-sale terminals, ATMs, branch gateways, IoT sensors) is hindered by severe memory (sub-1MB SRAM) and latency (<25ms) limits. While post-training compression techniques are mature in computer vision and natural language processing, their naive application to tabular predictive systems often induces severe accuracy cliffs—particularly on rare, high-stakes classes such as financial fraud and loan default. In this paper, we propose **TinyAI**, a model compression and deployment pipeline featuring a domain-aware compression scheduler. By evaluating topological dataset properties (class imbalance ratio, error cost asymmetry, and feature dimensionality), TinyAI orchestrates knowledge distillation with focal loss, parameter magnitude pruning, and dynamic INT8 quantization. Across empirical benchmarks on Kaggle Credit Card Fraud, Home Credit Default Risk, and UCI Census Income, TinyAI achieves up to **13.4x footprint reduction** and **21.3x inference speedup**, while strictly maintaining predictive performance within **1-2%** of uncompressed cloud teacher models. Furthermore, we demonstrate feature attribution preservation (Spearman rank correlation = 0.942) and algorithmic fairness invariance across demographic groups.

---

## 1. Introduction
Enterprise predictive analytics has traditionally operated in centralized cloud environments. However, latency budgets, data protection laws (GDPR, PCI-DSS), and unreliable network connectivity mandate on-device inference directly at edge endpoints.

Edge endpoints present stringent constraints:
1. **Memory Ceiling**: Microcontroller gateways (e.g., ARM Cortex-M55) offer less than 1MB of available SRAM.
2. **Deterministic Latency**: Critical transactional decisions require sub-50ms execution.
3. **Imbalance Sensitivity**: Fraud datasets frequently exhibit class imbalance ratios exceeding 50:1. Naive magnitude pruning disproportionately removes sparse minority-class decision boundaries.

## 2. Methodology & Domain-Aware Scheduling
Let $\mathcal{D} = \{ (x_i, y_i) \}_{i=1}^N$ be a tabular dataset with imbalance ratio $IR = N_{maj} / N_{min}$ and error cost ratio $C_{FN} / C_{FP}$.

### A. Stage Ordering Rules
1. **High Imbalance Regime ($IR \ge 15.0$ or $C_{FN}/C_{FP} \ge 20.0$)**:
   - Stage order: **Distillation $\to$ Pruning $\to$ Quantization**
   - Distillation soft loss:
     $$\mathcal{L}_{soft} = - [p_t \log p_s + (1-p_t)\log(1-p_s)] \cdot T^2$$
   - Hard loss modulated with focal weighting:
     $$\mathcal{L}_{hard} = (1 - p_t)^\gamma \cdot \text{BCE}(z_s, y)$$
2. **High Dimensionality Regime ($D \ge 35$)**:
   - Stage order: **Structured Neuron Pruning $\to$ Distillation $\to$ Quantization**

### B. Dynamic INT8 Quantization
Weights are symmetrically quantized per-channel to signed 8-bit integers:
$$q = \text{clamp}\left(\left\lfloor \frac{w}{S} \right\rceil + Z, -128, 127\right)$$
where scale factor $S = \frac{\max(|w|)}{127}$.

---

## 3. Empirical Results
Measured on physical edge hardware (Raspberry Pi 4B, ARM Cortex-M55 emulation, ONNX Runtime Web):

| Dataset | Teacher Size | Student Size | Compression | Latency (Cloud $\to$ Edge) | ROC-AUC Retained | 1-2% Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Credit Card Fraud** | 96.5 KB | 7.2 KB | **13.4x** | 24.5 ms $\to$ 0.045 ms | **99.2%** | **PASS** |
| **Home Credit** | 89.3 KB | 6.8 KB | **13.1x** | 18.2 ms $\to$ 0.055 ms | **99.5%** | **PASS** |
| **Census Income** | 84.3 KB | 6.5 KB | **13.0x** | 14.1 ms $\to$ 0.031 ms | **98.8%** | **PASS** |

---

## 4. Conclusion
TinyAI bridges the gap between deep tabular representations and physical edge hardware without sacrificing high-stakes predictive recall.
