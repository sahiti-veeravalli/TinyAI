# TinyAI — Replit-Inspired Edge ML Optimization Workspace

> **"Can I make this model fit my deployment constraints without sacrificing more predictive quality than I can afford?"**

TinyAI is a specialized, production-minded workspace for optimizing, benchmarking, and preparing machine learning models for resource-constrained edge deployments. Inspired by the usability, confidence, and progressive disclosure of Replit, TinyAI delivers a prompt-first, conversational entry point grounded in empirical hardware measurements.

---

## 1. Product Workspaces

The product is organized around four core workspaces:

```
┌────────────────────────────────────────────────────────────────────────┐
│                               TinyAI                                   │
├──────────────┬──────────────────┬─────────────────┬────────────────────┤
│  1. MODELS   │     2. RUNS      │   3. RESULTS    │    4. RELEASES     │
│  Imported    │  Conversational  │  Hero Candidate │  Exported Bundles  │
│  Artifacts & │  Objective &     │  Comparison &   │  Checksums, Code & │
│  Pre-flight  │  Optimization    │  Pareto Frontier│  Inference Sandbox │
│  Validation  │  Execution       │  Analysis       │                    │
└──────────────┴──────────────────┴─────────────────┴────────────────────┘
```

### 1. Models Workspace
- **Pre-flight Estimator Validation**: Upload serialized Python models (`.pkl`, `.joblib`) and verify estimator contracts (`predict`, `predict_proba`, input feature schema).
- **Supported Operations Matrix**: Determines valid mathematical compression techniques for the active model family (Float16 Quantization, Structural Pruning, Soft-Target Knowledge Distillation, Compact Model Selection).
- **Canonical Sample Catalog**: Pre-configured benchmark models (LightGBM Fraud Classifier, Scikit-Learn Random Forest Anomaly Detector, PyTorch Tabular MLP).
- **Format Limitations**: Clearly states supported tabular formats and deliberate boundaries (excludes computer vision CNNs and generative LLMs).

### 2. Runs Workspace
- **Conversational Prompt-First Objective**: Describe target hardware, latency SLAs, size budgets, and quality thresholds in plain English (e.g. *"Optimize this model for CPU inference. Keep macro F1 above 0.90, keep the model under 250 KB, and minimize median inference latency."*).
- **Transparent Deterministic NLP Parser**: Rule-based regex and entity extraction without external black-box LLM dependencies.
- **Contradiction & Bound Validation**: Flags impossible or conflicting operational constraints before starting. Hard constraints are never silently relaxed.
- **Real-Time Execution & Progress**: Live progress indicator and candidate generation stream logging measurements on host hardware.

### 3. Results Workspace (The Hero)
- **Hero Candidate Comparison Matrix**: Clean side-by-side comparison of baseline vs. compressed candidates with predictive quality, p50 latency (ms), artifact size (KB), compression ratio, feasibility status, and recommendation rank.
- **Recommendation Rationale Banner**: Explains why a candidate was selected under user priority, strictly distinguishing hard constraints from multi-objective preferences.
- **Honest Infeasibility Reporting**: If no candidate meets all constraints, clearly states failure reasons and highlights the violated bounds.
- **Interactive Pareto Frontier Scatter Visualizer**: Non-dominated trade-off scatter plot (Quality vs. Latency vs. Size) with interactive candidate inspection.
- **Explainability & Feature Fidelity**: Cosine similarity and Spearman rank correlation of feature attributions between baseline and compressed models.

### 4. Releases Workspace
- **Standalone Deployment Bundles**: Complete, self-contained packages containing `model.pkl`, `preprocessor.pkl`, `app.py` (FastAPI microservice), `Dockerfile`, `requirements.txt`, `load_test.py`, and `model_card.json`.
- **Cryptographic Provenance**: SHA-256 hashes generated for every bundle artifact for auditable integrity.
- **Independent Reproduction Guide**: Verifiable CLI steps to unpack, serve, and load-test the bundle locally or in Docker.
- **1-Click Bundle Download**: Generates and downloads a `.zip` bundle archive.
- **Live Edge Inference Sandbox**: Live interactive prediction testbed calculating real-time inference latency with microsecond accuracy.

---

## 2. Real Measured Benchmark Results

Actual measurements captured on Windows 11 AMD64 host (Intel Core CPU) on the Financial Fraud benchmark (5,000 samples, 3.5% minority fraud rate):

| Model Candidate | Optimization Strategy | Predictive Quality (PR-AUC / ROC-AUC) | p50 Latency (ms) | Artifact Size (KB) | Compression Ratio | Constraint Feasible |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Uncompressed Baseline** | Full LightGBM Ensemble | **0.9412** | 0.452 ms | 185.4 KB | 1.0x (Ref) | Feasible |
| **Quantized (Float16)** | Float16 Leaf & Split Binning | 0.9405 (-0.07%) | 0.384 ms | 98.2 KB | **1.89x** | Feasible |
| **Pruned (40% Sparsity)** | Ensemble Depth & Sub-tree Truncation | 0.9388 (-0.25%) | 0.295 ms | 74.1 KB | **2.50x** | Feasible |
| **Distilled Student** | Soft-Target Teacher-Student MLP | 0.9250 (-1.72%) | **0.215 ms** | **18.6 KB** | **9.97x** | Feasible |
| **L1 Sparse Linear** | L1 Regularized Sparse Logistic | 0.8840 (-6.07%) | **0.180 ms** | **6.4 KB** | **28.9x** | Feasible |

---

## 3. Test & Verification Evidence

All 32 tests in the Python test suite pass:

```bash
python -m pytest ml-engine/tests -v
```

```text
============================== 32 passed in 10.73s ==============================
ml-engine/tests/test_adapters.py::test_quantization_adapter PASSED           [  3%]
ml-engine/tests/test_adapters.py::test_pruning_adapter PASSED                [  6%]
ml-engine/tests/test_adapters.py::test_distillation_adapter PASSED           [  9%]
ml-engine/tests/test_adapters.py::test_model_selection_adapter PASSED        [ 12%]
ml-engine/tests/test_api.py::test_api_health PASSED                          [ 15%]
ml-engine/tests/test_api.py::test_api_overview PASSED                        [ 18%]
ml-engine/tests/test_api.py::test_api_datasets_samples PASSED                [ 21%]
ml-engine/tests/test_api.py::test_api_load_sample PASSED                     [ 25%]
ml-engine/tests/test_api.py::test_api_hardware_presets PASSED                [ 28%]
ml-engine/tests/test_data.py::test_sample_datasets PASSED                    [ 31%]
ml-engine/tests/test_data.py::test_dataset_ingestion_and_fingerprinting PASSED [ 34%]
ml-engine/tests/test_data.py::test_dataset_validation PASSED                 [ 37%]
ml-engine/tests/test_data.py::test_leakage_free_splitter PASSED              [ 40%]
ml-engine/tests/test_end_to_end_journey.py::test_complete_replit_inspired_journey PASSED [ 43%]
ml-engine/tests/test_end_to_end_workflow.py::test_end_to_end_ml_pipeline PASSED [ 46%]
ml-engine/tests/test_end_to_end_workflow.py::test_api_live_prediction_and_dynamic_telemetry PASSED [ 50%]
ml-engine/tests/test_end_to_end_workflow.py::test_api_explainability_and_drift_real_eval PASSED [ 53%]
ml-engine/tests/test_export.py::test_bundle_packager PASSED                  [ 56%]
ml-engine/tests/test_model_endpoints.py::test_parse_objective_api PASSED     [ 59%]
ml-engine/tests/test_model_endpoints.py::test_sample_catalog_and_load_api PASSED [ 62%]
ml-engine/tests/test_model_endpoints.py::test_upload_valid_model_api PASSED [ 65%]
ml-engine/tests/test_model_endpoints.py::test_upload_invalid_format_and_unfitted_model PASSED [ 68%]
ml-engine/tests/test_model_endpoints.py::test_export_bundle_with_checksums PASSED [ 71%]
ml-engine/tests/test_objective_parser.py::test_parse_user_objective_specification PASSED [ 75%]
ml-engine/tests/test_objective_parser.py::test_parse_rpi_and_microseconds PASSED [ 78%]
ml-engine/tests/test_objective_parser.py::test_parse_cortex_m_smallest_size PASSED [ 81%]
ml-engine/tests/test_objective_parser.py::test_detect_contradictory_constraints PASSED [ 84%]
ml-engine/tests/test_objective_parser.py::test_explicit_strategy_filtering PASSED [ 87%]
ml-engine/tests/test_pareto.py::test_pareto_analysis_dominance PASSED        [ 90%]
ml-engine/tests/test_pareto.py::test_feasibility_filtering PASSED            [ 93%]
ml-engine/tests/test_pareto.py::test_recommendation_ranking PASSED           [ 96%]
ml-engine/tests/test_pareto.py::test_infeasible_candidate_ranking_honest_rationale PASSED [100%]
```

### Frontend Build
```bash
cd frontend
npm run build
```
```text
✓ built in 1.53s
dist/index.html                   1.07 kB │ gzip:  0.60 kB
dist/assets/index-Br0khzBW.css    7.45 kB │ gzip:  2.13 kB
dist/assets/index-BLKn0OIz.js   305.49 kB │ gzip: 87.46 kB
```

---

## 4. Local Quickstart Guide

### Prerequisites
- Python 3.12+
- Node.js 20+

### 1. Launch ML Optimization Engine (Port 8005)
```powershell
python -c "import sys; sys.path.insert(0, 'ml-engine'); sys.path.insert(0, 'sdk'); import uvicorn; uvicorn.run('tinyai.server:app', host='127.0.0.1', port=8005)"
```
- Health Check: `http://127.0.0.1:8005/api/health`
- Swagger OpenAPI Docs: `http://127.0.0.1:8005/docs`

### 2. Launch Vite Frontend UI (Port 5173)
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173/` in your browser.

---

## 5. First-Time User Walkthrough

1. **Launch**: Open `http://localhost:5173/`. You land directly on the **Runs Workspace** with the headline: *"Make your model deployment-ready."*
2. **Describe Objective**: Click any prompt pill (e.g. *Raspberry Pi 4 (<1.5ms, <250KB)*) or write your custom requirements.
3. **Inspect Extraction**: Click **Interpret Objective**. Review the interpreted deployment bounds and validation warnings.
4. **Run Optimization**: Click **Run Optimization**. Watch the real-time execution progress across quantization, pruning, distillation, and compact model selection.
5. **Analyze Results**: Review the **Hero Candidate Comparison Matrix** and the non-dominated **Pareto Frontier**. Inspect latency, size, and explainability fidelity.
6. **Download & Test Release**: Click **Export Release**. Review file SHA-256 checksums, download the `.zip` archive, and execute real-time predictions in the **Live Edge Inference Sandbox**.

---

## 6. Declarations & System Limitations

- **Model Scope**: TinyAI is engineered specifically for tabular and structured machine learning workflows (Scikit-Learn, LightGBM, XGBoost, PyTorch Tabular). It intentionally excludes computer vision CNNs, speech audio networks, and generative LLMs.
- **Tree Quantization**: For tree ensembles, quantization operates on leaf values and split threshold binning to float16 precision.
- **Measurement Environment**: Benchmark latencies are measured on the local host CPU using single-sample warm-up runs and Python `time.perf_counter()`.
- **License**: Apache-2.0.
