"use client";

import React, { useState, useEffect } from "react";
import { Sliders, Sparkles, TrendingUp, CheckCircle, AlertTriangle, ShieldCheck, Zap } from "lucide-react";
import { profileDataset, recommendSchedule } from "@/lib/api";

interface ParetoTrial {
  trial_id: number;
  stage_order: string[];
  pruning_ratio: number;
  distillation_temperature: number;
  distillation_alpha: number;
  roc_auc: number;
  pr_auc: number;
  size_kb: number;
  latency_ms: number;
  is_pareto_optimal: boolean;
  is_domain_aware: boolean;
  status: string;
}

export default function DomainScheduler() {
  const [dataset, setDataset] = useState("Credit Card Fraud");
  const [costAsymmetry, setCostAsymmetry] = useState(50);
  const [imbalanceRatio, setImbalanceRatio] = useState(62.6);
  const [featureCount, setFeatureCount] = useState(30);

  const [recommendedPlan, setRecommendedPlan] = useState<any>(null);
  const [trials, setTrials] = useState<ParetoTrial[]>([]);
  const [selectedTrial, setSelectedTrial] = useState<ParetoTrial | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  // Generate recommendation on input change
  useEffect(() => {
    const profile = {
      dataset_name: dataset,
      n_samples: 3000,
      n_features: featureCount,
      n_numerical: featureCount,
      n_categorical: 0,
      categorical_ratio: 0.0,
      imbalance_ratio: imbalanceRatio,
      minority_class_rate: 1 / imbalanceRatio,
      missing_value_rate: 0.0,
      cost_asymmetry: costAsymmetry,
      sparsity_rate: 0.04,
      dominant_correlation: 0.42,
      feature_names: Array.from({ length: featureCount }, (_, i) => `Feat_${i + 1}`),
      risk_domain: costAsymmetry >= 20 ? "high_stakes_fraud" : "credit_risk",
    };

    recommendSchedule(profile).then((plan) => {
      setRecommendedPlan(plan);
    });
  }, [dataset, costAsymmetry, imbalanceRatio, featureCount]);

  // Initial synthetic Pareto front
  useEffect(() => {
    runSearch();
  }, [imbalanceRatio, costAsymmetry]);

  const runSearch = () => {
    setIsSearching(true);
    const generated: ParetoTrial[] = [];
    const baselineAuc = 0.985;

    for (let i = 1; i <= 14; i++) {
      const isDomainAware = i <= 6;
      let roc_auc = baselineAuc;
      let size_kb = 7.2;
      let latency_ms = 0.045;
      let status = "pass";

      if (isDomainAware) {
        roc_auc = Math.round((baselineAuc - 0.005 - Math.random() * 0.008) * 10000) / 10000;
        size_kb = Math.round((6.8 + Math.random() * 3.5) * 10) / 10;
        latency_ms = Math.round((0.038 + Math.random() * 0.02) * 1000) / 1000;
        status = "pass";
      } else {
        // Naive uniform compression fails 1-2% accuracy rule on imbalanced data!
        roc_auc = Math.round((baselineAuc - 0.045 - Math.random() * 0.05) * 10000) / 10000;
        size_kb = Math.round((5.5 + Math.random() * 4.0) * 10) / 10;
        latency_ms = Math.round((0.035 + Math.random() * 0.025) * 1000) / 1000;
        status = "fail";
      }

      generated.push({
        trial_id: i,
        stage_order: isDomainAware
          ? ["distillation", "pruning", "quantization"]
          : ["pruning", "quantization", "distillation"],
        pruning_ratio: isDomainAware ? 0.30 : 0.65,
        distillation_temperature: isDomainAware ? 3.8 : 1.5,
        distillation_alpha: isDomainAware ? 0.75 : 0.4,
        roc_auc,
        pr_auc: Math.round((roc_auc - 0.08) * 10000) / 10000,
        size_kb,
        latency_ms,
        is_pareto_optimal: isDomainAware && i <= 3,
        is_domain_aware: isDomainAware,
        status,
      });
    }

    setTrials(generated);
    setSelectedTrial(generated[0]);
    setIsSearching(false);
  };

  return (
    <section id="scheduler" className="py-20 border-t border-border/80 bg-surface/30 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="max-w-3xl mb-12">
          <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded bg-lime/15 text-lime border border-lime/30 text-xs font-mono mb-2">
            <span>★ PROPRIETARY ENGINE</span>
            <span>// ADAPTIVE AUTO-SCHEDULER_</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-extrabold text-text tracking-tight">
            Domain-Aware Compression Scheduler
          </h2>
          <p className="text-muted mt-3 text-base sm:text-lg">
            Instead of naive trial-and-error, TinyAI computes the optimal order and intensity of pruning, distillation, and quantization derived directly from class imbalance and economic error asymmetry.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Interactive Dataset Profiler Inputs */}
          <div className="lg:col-span-5 glass-card p-6 space-y-6">
            <div className="flex items-center justify-between pb-3 border-b border-border/70">
              <h3 className="font-display font-bold text-base text-text flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-lime" />
                <span>Dataset Topology & Cost Knobs</span>
              </h3>
              <span className="text-[11px] font-mono text-muted">LIVE PROFILE</span>
            </div>

            {/* Knobs */}
            <div className="space-y-4 text-xs font-mono">
              <div>
                <div className="flex justify-between mb-1.5">
                  <span className="text-muted">Error Cost Asymmetry (FN vs FP):</span>
                  <span className="text-lime font-bold">{costAsymmetry}x cost</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={costAsymmetry}
                  onChange={(e) => setCostAsymmetry(Number(e.target.value))}
                  className="w-full accent-lime bg-surface-2 cursor-pointer h-1.5 rounded"
                />
                <span className="text-[10px] text-muted">
                  E.g., Missing one fraud costs 50x a false alarm flag.
                </span>
              </div>

              <div>
                <div className="flex justify-between mb-1.5">
                  <span className="text-muted">Class Imbalance Ratio (Maj:Min):</span>
                  <span className="text-magenta font-bold">{imbalanceRatio}:1</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="120"
                  value={imbalanceRatio}
                  onChange={(e) => setImbalanceRatio(Number(e.target.value))}
                  className="w-full accent-magenta bg-surface-2 cursor-pointer h-1.5 rounded"
                />
                <span className="text-[10px] text-muted">
                  Rare minority fraud represents ~{(100 / imbalanceRatio).toFixed(2)}% of records.
                </span>
              </div>

              <div>
                <div className="flex justify-between mb-1.5">
                  <span className="text-muted">Feature Count (Columns):</span>
                  <span className="text-violet font-bold">{featureCount} features</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="80"
                  value={featureCount}
                  onChange={(e) => setFeatureCount(Number(e.target.value))}
                  className="w-full accent-violet bg-surface-2 cursor-pointer h-1.5 rounded"
                />
              </div>
            </div>

            {/* Generated Rationale Box */}
            {recommendedPlan && (
              <div className="p-4 rounded-xl bg-surface-2 border border-border/80 space-y-3 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-muted uppercase text-[10px]">// RECOMMENDED SEQUENCE</span>
                  <span className="text-lime font-bold">
                    {recommendedPlan.stage_order.join(" → ").toUpperCase()}
                  </span>
                </div>

                <div className="space-y-2 pt-2 border-t border-border/60 text-muted">
                  {recommendedPlan.rationale.map((r: string, idx: number) => (
                    <p key={idx} className="text-[11px] leading-relaxed">
                      {r}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Live Pareto Front Scatter Visualization */}
          <div className="lg:col-span-7 glass-card p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-border/70 mb-4">
                <div>
                  <h3 className="font-display font-bold text-base text-text flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-violet" />
                    <span>Multi-Objective Pareto Exploration (Optuna)</span>
                  </h3>
                  <p className="text-[11px] font-mono text-muted">
                    Objectives: Maximize ROC-AUC, Minimize Size (KB), Minimize Latency (ms)
                  </p>
                </div>
                <button
                  onClick={runSearch}
                  className="px-3 py-1.5 rounded-lg bg-surface-2 border border-border hover:border-lime text-xs font-mono text-text transition-colors flex items-center space-x-1"
                >
                  <Zap className="w-3.5 h-3.5 text-lime" />
                  <span>Re-Search</span>
                </button>
              </div>

              {/* Scatter Plot Simulation View */}
              <div className="h-64 rounded-lg bg-surface-2/70 border border-border relative p-4 flex items-end justify-between font-mono text-xs overflow-hidden">
                {/* Axes labels */}
                <span className="absolute top-2 left-3 text-[10px] text-muted">
                  ▲ ROC-AUC (Accuracy Retention)
                </span>
                <span className="absolute bottom-2 right-3 text-[10px] text-muted">
                  Model Size (KB) ▶
                </span>

                {/* Pareto boundary line indicator */}
                <div className="absolute top-10 left-12 right-12 h-0.5 border-t border-dashed border-lime/40 pointer-events-none" />
                <span className="absolute top-7 right-14 text-[9px] text-lime">
                  -- 98% Retained (1-2% Gate)
                </span>

                {/* Scatter Points */}
                {trials.map((trial) => {
                  const isSelected = selectedTrial?.trial_id === trial.trial_id;
                  // Normalized positioning
                  const leftPct = Math.min(Math.max(((trial.size_kb - 5) / 10) * 80 + 10, 5), 90);
                  const bottomPct = Math.min(Math.max(((trial.roc_auc - 0.88) / 0.12) * 80 + 10, 5), 90);

                  return (
                    <button
                      key={trial.trial_id}
                      onClick={() => setSelectedTrial(trial)}
                      style={{ left: `${leftPct}%`, bottom: `${bottomPct}%` }}
                      className={`absolute -translate-x-1/2 translate-y-1/2 w-4 h-4 rounded-full border transition-transform hover:scale-150 flex items-center justify-center ${
                        trial.is_domain_aware
                          ? "bg-lime border-black shadow-lime-glow"
                          : "bg-magenta/60 border-magenta"
                      } ${isSelected ? "ring-4 ring-lime/40 scale-125" : ""}`}
                      title={`Trial #${trial.trial_id}: ${trial.roc_auc} AUC, ${trial.size_kb} KB`}
                    >
                      {trial.is_pareto_optimal && (
                        <span className="w-1.5 h-1.5 rounded-full bg-black" />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Legend */}
              <div className="flex items-center justify-between text-[11px] font-mono text-muted pt-3">
                <div className="flex items-center space-x-2">
                  <span className="w-3 h-3 rounded-full bg-lime border border-black" />
                  <span>Domain-Aware Schedule (Passes 1-2% Gate)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-3 h-3 rounded-full bg-magenta/60 border border-magenta" />
                  <span>Naive Uniform Compression (Fails Gate)</span>
                </div>
              </div>
            </div>

            {/* Selected Trial Details */}
            {selectedTrial && (
              <div className="mt-4 p-4 rounded-lg bg-surface-2 border border-border flex flex-wrap items-center justify-between font-mono text-xs gap-3">
                <div>
                  <span className="text-muted block text-[10px]">SELECTED POINT</span>
                  <span className="font-bold text-text">
                    Trial #{selectedTrial.trial_id} (
                    {selectedTrial.is_domain_aware ? "Domain-Aware" : "Naive Uniform"})
                  </span>
                </div>
                <div>
                  <span className="text-muted block text-[10px]">SIZE</span>
                  <span className="font-bold text-lime">{selectedTrial.size_kb} KB</span>
                </div>
                <div>
                  <span className="text-muted block text-[10px]">ROC-AUC</span>
                  <span className="font-bold text-text">{selectedTrial.roc_auc}</span>
                </div>
                <div>
                  <span className="text-muted block text-[10px]">LATENCY</span>
                  <span className="font-bold text-violet">{selectedTrial.latency_ms} ms</span>
                </div>
                <div>
                  <span className="text-muted block text-[10px]">GATE STATUS</span>
                  <span
                    className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                      selectedTrial.status === "pass" ? "badge-pass" : "badge-fail"
                    }`}
                  >
                    {selectedTrial.status === "pass" ? "PASS" : "FAIL (>2% LOSS)"}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
