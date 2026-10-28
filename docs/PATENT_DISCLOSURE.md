# Patent Disclosure: Domain-Aware Compression Schedule for Edge AI in Tabular Predictive Systems

**Filing Entity**: TinyAI Research Labs  
**Inventor(s)**: Sahiti Veeravalli et al.  
**Classification**: G06N 3/08 (Learning methods), G06F 17/10 (Complex mathematical operations)  

---

## 1. Field of the Invention
The present invention relates generally to artificial intelligence and edge computing, and more specifically to automated, dataset-conditioned methods and systems for compressing tabular neural networks and tree ensembles for execution on resource-constrained embedded devices.

---

## 2. Background and Prior Art Limitations
Existing model compression toolkits (such as standard PyTorch pruning utilities, TensorFlow Lite, and generic post-training quantization tools) operate using static, heuristic pipelines. For instance, post-training quantization is uniformly applied after full model convergence. 

**The Technical Problem**: In tabular predictive systems characterized by high class imbalance (e.g. financial fraud where anomalies represent <1% of transactions) and asymmetric error costs, conventional magnitude pruning zero-outs rare-class decision manifolds. Consequently, conventional pipelines exhibit catastrophic drops in Recall at fixed High Precision (e.g., Recall drops by >35%).

---

## 3. Summary of the Invention & Novel Claims

### Claim 1: A Domain-Aware Compression Scheduling System
A computer-implemented method for compressing a predictive machine learning model for edge deployment, comprising:
1. **Profiling** an input tabular dataset to determine a multidimensional profile comprising:
   - a class imbalance ratio ($IR = N_{maj} / N_{min}$);
   - an error cost asymmetry factor ($C_{FN} / C_{FP}$);
   - a feature sparsity and dimensionality factor;
2. **Dynamically Selecting a Compression Pipeline Sequence** from a plurality of permutation orders of pruning, knowledge distillation, and quantization, wherein:
   - When the class imbalance ratio exceeds a predetermined threshold ($IR \ge 15.0$) or the cost asymmetry factor exceeds a predetermined threshold ($C_{FN} / C_{FP} \ge 20.0$), the system enforces **Distillation Prior to Weight Pruning**, wherein knowledge distillation employs temperature-scaled soft targets and focal loss weighting;
   - When feature dimensionality exceeds a predetermined count ($D \ge 35$), the system enforces **Structured Neuron Pruning Prior to Distillation**;
3. **Executing Dynamic Quantization** on the resultant pruned student graph to generate an ONNX binary conforming to physical edge SRAM and latency budgets.

---

## 4. Prior Art Comparison Table

| Attribute | Prior Art (Generic PTQ / Pruning) | TinyAI Domain-Aware Scheduler |
| :--- | :--- | :--- |
| **Pipeline Order** | Fixed (Train $\to$ Prune $\to$ Quantize) | **Dynamic Permutation conditioned on Dataset Topology** |
| **Class Imbalance Awareness** | None (Uniform parameter zeroing) | **Focal Loss Distillation prior to pruning** |
| **Asymmetric Risk Costs** | Ignored | **Dynamically modulates pruning budget $\tau$** |
| **Recall Retention** | Drops by 25% - 40% on rare anomalies | **Maintains Recall within 1-2% of Cloud Teacher** |

---

*Disclaimer: This document is an intellectual property disclosure summary prepared for research demonstration and patent drafting purposes; it does not constitute formal legal counsel.*
