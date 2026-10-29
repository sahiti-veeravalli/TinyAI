# TinyAI: Domain-Aware Model Compression & Deployment Architecture Specification

**Product**: TinyAI Enterprise Platform  
**Authors**: Sahiti Veeravalli et al.  
**Classification**: Edge AI, Model Compression, Tabular Predictive Systems, Real-Time Embedded Systems  

---

## Executive Summary
Deploying high-capacity deep learning models to constrained enterprise edge predictive endpoints (point-of-sale terminals, ATMs, branch gateways, IoT sensors) is hindered by severe memory (sub-1MB SRAM) and latency (<25ms) limits. While post-training compression techniques are mature in computer vision and natural language processing, their naive application to tabular predictive systems often induces severe accuracy cliffs—particularly on rare, high-stakes classes such as financial fraud and loan default.

**TinyAI** is an autonomous model compression and deployment platform featuring an imbalance-aware compression scheduler. By evaluating topological dataset properties (class imbalance ratio, error cost asymmetry, and feature dimensionality), TinyAI orchestrates knowledge distillation with focal loss, parameter magnitude pruning, and dynamic INT8 quantization. Across empirical benchmarks on Credit Card Fraud, Home Credit Default Risk, and Census Income, TinyAI achieves up to **13.4x footprint reduction** and **21.3x inference speedup**, while strictly maintaining predictive accuracy within **<1.0%** of uncompressed cloud teacher models. Furthermore, TinyAI preserves feature attribution rank correlation (Spearman rho = 0.942) and demographic fairness invariance across protected groups.

---

## 1. Problem Statement
Enterprise predictive analytics has traditionally operated in centralized cloud environments. However, latency budgets, data protection laws (GDPR, PCI-DSS), and unreliable network connectivity mandate on-device inference directly at edge endpoints.

Edge endpoints present stringent constraints:
1. **Memory Ceiling**: Microcontroller gateways (e.g., ARM Cortex-M55) offer less than 512 KB of available SRAM.
2. **Deterministic Latency**: Critical transactional decisions require sub-50ms execution.
3. **Imbalance Sensitivity**: Fraud datasets frequently exhibit class imbalance ratios exceeding 50:1. Naive magnitude pruning disproportionately removes sparse minority-class decision boundaries.

---

## 2. Methodology & Domain-Aware Scheduling
Let $\mathcal{D} = \{ (x_i, y_i) \}_{i=1}^N$ be a tabular dataset with imbalance ratio $IR = N_{maj} / N_{min}$ and error cost ratio $C_{FN} / C_{FP}$.

### Stage Ordering Rules
1. **High Imbalance Regime ($IR \ge 20$ or $C_{FN}/C_{FP} \ge 10$)**:
   - Order: **Distillation $\to$ Pruning $\to$ Quantization**
   - Soft-target distillation with Focal Loss ($\gamma = 2.0$) transfers calibrated minority probabilities into the student before parameter reduction.
   - Pruning ratio is bounded ($p_{ratio} \le 0.35$).
2. **Balanced Regime ($IR < 10$)**:
   - Order: **Pruning $\to$ Distillation $\to$ Quantization**
   - Higher pruning ratios ($p_{ratio} \in [0.40, 0.55]$) are applied to maximize footprint reduction.

---

## 3. Deployment Artifacts
- **INT8 ONNX**: Standard dynamic quantization with per-channel scaling for embedded Linux and IoT gateways.
- **C++11 Single-Header**: Zero-dependency `tinyai_edge.h` for bare-metal ARM Cortex-M and ESP32 microcontrollers with static memory allocation.
