"use client";

import React, { useState } from "react";
import { Database, BrainCircuit, Scissors, Sparkles, Cpu, Download, Rocket, Activity, ChevronRight, X, Info } from "lucide-react";

interface StageInfo {
  id: string;
  name: string;
  short: string;
  color: string;
  icon: any;
  whatItDoes: string;
  mathFormula: string;
  parameters: Record<string, string>;
  beforeMetric: string;
  afterMetric: string;
  patentRole: string;
}

const STAGES: StageInfo[] = [
  {
    id: "data",
    name: "01. Dataset Profiler",
    short: "Profiler",
    color: "var(--violet)",
    icon: Database,
    whatItDoes: "Analyzes structural, topological, and cost characteristics of input tabular data.",
    mathFormula: "IR = N_{maj} / N_{min}, \\quad \\text{CostRatio} = C_{FN} / C_{FP}",
    parameters: {
      "Imbalance Ratio": "62.6:1 (Fraud)",
      "Cost Asymmetry": "50x (Missed Fraud vs False Alarm)",
      "Feature Count": "30 continuous components",
    },
    beforeMetric: "Raw CSV/Parquet dataset with skewed variance",
    afterMetric: "Normalized profile feeding Domain-Aware Scheduler",
    patentRole: "Extracts decision variables that dictate the compression sequence.",
  },
  {
    id: "teacher",
    name: "02. Teacher Model",
    short: "Teacher",
    color: "var(--violet)",
    icon: BrainCircuit,
    whatItDoes: "Trains a high-capacity Tabular ResNet MLP with skip connections and LayerNorm.",
    mathFormula: "h_{l+1} = \\text{SiLU}(\\text{LN}(W_l h_l + b_l)) + h_l",
    parameters: {
      "Hidden Dimensions": "(128, 64, 32)",
      "Loss": "Weighted BCE with pos_weight = min(cost, 20.0)",
      "Capacity": "Large representational space",
    },
    beforeMetric: "Untrained random initialization",
    afterMetric: "ROC-AUC: 1.0, Size: 96.5 KB (Heavy for microcontrollers)",
    patentRole: "Provides calibrated soft probability logit distributions for distillation.",
  },
  {
    id: "distill",
    name: "03. Knowledge Distillation",
    short: "Distillation",
    color: "var(--violet)",
    icon: Sparkles,
    whatItDoes: "Transfers rich decision manifolds into a compact 2-layer student MLP using temperature-scaled soft targets.",
    mathFormula: "\\mathcal{L}_{KD} = \\alpha T^2 \\text{BCE}(\\sigma(z_s/T), \\sigma(z_t/T)) + (1-\\alpha) (1-p_t)^\\gamma \\text{BCE}(z_s, y)",
    parameters: {
      "Temperature (T)": "3.8 (softens extreme probability peaks)",
      "Alpha (α)": "0.75 (teacher supervision weight)",
      "Focal Gamma (γ)": "2.0 (forces focus on hard edge fraud)",
    },
    beforeMetric: "Empty student architecture",
    afterMetric: "Retains 99.2% of teacher ROC-AUC in compact 32->16 hidden size",
    patentRole: "Crucial for imbalanced data: preserves rare minority boundaries prior to weight zeroing.",
  },
  {
    id: "prune",
    name: "04. Pruning Engine",
    short: "Pruning",
    color: "var(--magenta)",
    icon: Scissors,
    whatItDoes: "Removes redundant parameters via L1 unstructured magnitude pruning and bakes mask into weights.",
    mathFormula: "W_{pruned} = W \\odot \\mathbf{1}_{|W| > \\tau_{amount}}",
    parameters: {
      "Pruning Ratio": "30% - 45% (Domain-adaptive)",
      "Method": "Global L1 Unstructured Magnitude",
      "Mask Consolidation": "Permanent (Zero runtime mask overhead)",
    },
    beforeMetric: "Dense parameter matrices",
    afterMetric: "30% exact zeros baked into weights",
    patentRole: "Pruning ratio dynamically restrained when class imbalance is severe.",
  },
  {
    id: "quantize",
    name: "05. Dynamic INT8 Quantization",
    short: "Quantization",
    color: "var(--amber)",
    icon: Cpu,
    whatItDoes: "Converts 32-bit floating point weights into signed 8-bit integers with per-channel scaling.",
    mathFormula: "q = \\text{clamp}\\left(\\text{round}\\left(\\frac{w}{S}\\right) + Z, -128, 127\\right)",
    parameters: {
      "Precision": "Signed 8-bit Integer (QInt8)",
      "Scale Type": "Per-Channel Symmetric Weights",
      "Activations": "Dynamic runtime scaling",
    },
    beforeMetric: "FP32 Model: 96.5 KB",
    afterMetric: "INT8 Model: 7.2 KB (13.4x compression factor)",
    patentRole: "Preserves dynamic range on skewed distributions without outlier clipping.",
  },
  {
    id: "onnx",
    name: "06. ONNX Edge Export",
    short: "ONNX Export",
    color: "var(--lime)",
    icon: Download,
    whatItDoes: "Exports standardized computational graph compatible with ONNX Runtime Web and embedded C++ runtimes.",
    mathFormula: "\\text{Graph}(\\mathcal{O}) = \\text{FoldConstants}(\\text{FuseLinearAct}(\\mathcal{M}))",
    parameters: {
      "Opset Version": "14 / 18",
      "Target Execution": "WebGPU / WebAssembly / ARM CMSIS-NN",
    },
    beforeMetric: "Framework-dependent PyTorch state dictionary",
    afterMetric: "Zero-dependency single-file binary runtime",
    patentRole: "Enables instant deployment on any edge operating system.",
  },
];

export default function PipelineFlow() {
  const [selectedStage, setSelectedStage] = useState<StageInfo>(STAGES[2]); // Default Distillation

  return (
    <section id="pipeline" className="py-20 bg-surface/20 border-t border-border/80 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="max-w-3xl mb-12">
          <p className="text-xs font-mono uppercase text-violet tracking-wider mb-2">
            // 02 INTERACTIVE PIPELINE ARCHITECTURE_
          </p>
          <h2 className="font-display text-3xl sm:text-4xl font-extrabold text-text tracking-tight">
            The Domain-Aware Compression Flow
          </h2>
          <p className="text-muted mt-3 text-base sm:text-lg">
            Unlike fixed pipelines that prune blindly, TinyAI dynamically alters stage ordering and hyperparameter intensity based on dataset physics. Click any stage to inspect the mathematics and verified metrics.
          </p>
        </div>

        {/* Animated Sequence Flow Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-8">
          {STAGES.map((stage, idx) => {
            const Icon = stage.icon;
            const isSelected = selectedStage.id === stage.id;
            return (
              <button
                key={stage.id}
                onClick={() => setSelectedStage(stage)}
                className={`p-3.5 rounded-xl border text-left font-mono transition-all relative overflow-hidden flex flex-col justify-between h-28 ${
                  isSelected
                    ? "bg-surface-2 border-lime shadow-lime-glow"
                    : "bg-surface border-border hover:border-violet"
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <div
                    className="p-1.5 rounded-lg border border-border"
                    style={{ color: stage.color }}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-[10px] text-muted font-mono">0{idx + 1}</span>
                </div>
                <div>
                  <p className="text-xs font-bold text-text truncate">{stage.short}</p>
                  <p className="text-[10px] text-muted truncate">{stage.id.toUpperCase()}</p>
                </div>
                {isSelected && (
                  <div className="absolute bottom-0 left-0 right-0 h-1 bg-lime animate-pulse" />
                )}
              </button>
            );
          })}
        </div>

        {/* Side Detail Panel for Selected Stage */}
        <div className="glass-card p-6 sm:p-8 relative">
          <div className="flex items-start justify-between pb-4 border-b border-border/80 mb-6">
            <div className="flex items-center space-x-3">
              <div
                className="p-2.5 rounded-lg border border-border bg-surface-2"
                style={{ color: selectedStage.color }}
              >
                <selectedStage.icon className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-display font-extrabold text-xl sm:text-2xl text-text">
                  {selectedStage.name}
                </h3>
                <p className="text-xs font-mono text-muted">{selectedStage.whatItDoes}</p>
              </div>
            </div>
            <span
              className="px-2.5 py-1 rounded text-xs font-mono font-bold uppercase"
              style={{ backgroundColor: `${selectedStage.color}20`, color: selectedStage.color }}
            >
              ACTIVE STAGE
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Math Formula & Intuition */}
            <div className="lg:col-span-7 space-y-5">
              <div>
                <h4 className="text-xs font-mono uppercase text-muted mb-2">// MATHEMATICAL FORMULATION_</h4>
                <div className="p-4 rounded-lg bg-surface-2 border border-border/80 font-mono text-sm text-lime overflow-x-auto">
                  <code>{selectedStage.mathFormula}</code>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-mono uppercase text-muted mb-2">// PIPELINE ARCHITECTURAL ROLE_</h4>
                <div className="p-4 rounded-lg bg-violet/10 border border-violet/30 font-sans text-sm text-text leading-relaxed">
                  {selectedStage.patentRole}
                </div>
              </div>
            </div>

            {/* Parameters & Metrics */}
            <div className="lg:col-span-5 space-y-4">
              <h4 className="text-xs font-mono uppercase text-muted mb-2">// STAGE HYPERPARAMETERS_</h4>
              <div className="space-y-2 font-mono text-xs">
                {Object.entries(selectedStage.parameters).map(([key, val]) => (
                  <div key={key} className="flex items-center justify-between p-2.5 rounded bg-surface-2/60 border border-border/60">
                    <span className="text-muted">{key}:</span>
                    <span className="text-text font-bold text-right">{val}</span>
                  </div>
                ))}
              </div>

              <div className="p-3.5 rounded-lg bg-surface-2/80 border border-border/80 space-y-2 text-xs font-mono">
                <div>
                  <span className="text-muted block text-[10px] uppercase">// INPUT STATE</span>
                  <span className="text-magenta">{selectedStage.beforeMetric}</span>
                </div>
                <div>
                  <span className="text-muted block text-[10px] uppercase">// OUTPUT STATE</span>
                  <span className="text-lime">{selectedStage.afterMetric}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
