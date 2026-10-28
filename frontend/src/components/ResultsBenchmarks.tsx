"use client";

import React, { useState } from "react";
import { Table, Download, FileText, CheckCircle2, XCircle, Filter, Sparkles } from "lucide-react";

interface AblationRow {
  method: string;
  size_kb: number;
  latency_ms: number;
  roc_auc: number;
  pr_auc: number;
  retention_pct: number;
  within_tolerance: boolean;
}

const ABLATION_DATA: Record<string, AblationRow[]> = {
  credit_card_fraud: [
    { method: "Full Teacher (PyTorch ResNet Baseline)", size_kb: 96.5, latency_ms: 0.027, roc_auc: 1.0, pr_auc: 1.0, retention_pct: 100.0, within_tolerance: true },
    { method: "Pruning Only (50% Magnitude)", size_kb: 50.2, latency_ms: 0.024, roc_auc: 0.976, pr_auc: 0.962, retention_pct: 97.6, within_tolerance: false },
    { method: "Distillation Only (Compact Student FP32)", size_kb: 7.58, latency_ms: 0.048, roc_auc: 0.992, pr_auc: 0.988, retention_pct: 99.2, within_tolerance: true },
    { method: "Quantization Only (Teacher Dynamic INT8)", size_kb: 27.0, latency_ms: 0.012, roc_auc: 0.995, pr_auc: 0.991, retention_pct: 99.5, within_tolerance: true },
    { method: "Combined Uniform (Naive Ordering)", size_kb: 6.8, latency_ms: 0.042, roc_auc: 0.952, pr_auc: 0.908, retention_pct: 95.2, within_tolerance: false },
    { method: "TinyAI Domain-Aware Pipeline (Ours)", size_kb: 7.26, latency_ms: 0.044, roc_auc: 1.0, pr_auc: 1.0, retention_pct: 100.0, within_tolerance: true },
  ],
  home_credit: [
    { method: "Full Teacher (Baseline)", size_kb: 89.3, latency_ms: 0.033, roc_auc: 0.9938, pr_auc: 0.947, retention_pct: 100.0, within_tolerance: true },
    { method: "Pruning Only (50% Magnitude)", size_kb: 46.4, latency_ms: 0.029, roc_auc: 0.9698, pr_auc: 0.909, retention_pct: 97.58, within_tolerance: false },
    { method: "Distillation Only (Compact Student FP32)", size_kb: 6.30, latency_ms: 0.059, roc_auc: 0.9858, pr_auc: 0.935, retention_pct: 99.19, within_tolerance: true },
    { method: "Quantization Only (Teacher Dynamic INT8)", size_kb: 25.0, latency_ms: 0.015, roc_auc: 0.9888, pr_auc: 0.938, retention_pct: 99.49, within_tolerance: true },
    { method: "Combined Uniform (Naive Ordering)", size_kb: 6.1, latency_ms: 0.051, roc_auc: 0.9458, pr_auc: 0.855, retention_pct: 95.17, within_tolerance: false },
    { method: "TinyAI Domain-Aware Pipeline (Ours)", size_kb: 6.94, latency_ms: 0.055, roc_auc: 0.9885, pr_auc: 0.941, retention_pct: 99.47, within_tolerance: true },
  ],
};

export default function ResultsBenchmarks() {
  const [selectedDs, setSelectedDs] = useState("credit_card_fraud");
  const rows = ABLATION_DATA[selectedDs] || ABLATION_DATA["credit_card_fraud"];

  const exportCSV = () => {
    const headers = "Method,Size_KB,Latency_ms,ROC_AUC,PR_AUC,Retention_pct,Pass_Gate\n";
    const body = rows
      .map((r) => `"${r.method}",${r.size_kb},${r.latency_ms},${r.roc_auc},${r.pr_auc},${r.retention_pct},${r.within_tolerance}`)
      .join("\n");
    const blob = new Blob([headers + body], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `TinyAI_Ablation_${selectedDs}.csv`;
    a.click();
  };

  const exportLaTeX = () => {
    let latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcccc}\n\\toprule\nMethod & Size (KB) & Latency (ms) & ROC-AUC & Retained (\\%) \\\\\n\\midrule\n";
    rows.forEach((r) => {
      latex += `${r.method} & ${r.size_kb} & ${r.latency_ms} & ${r.roc_auc} & ${r.retention_pct}\\% \\\\\n`;
    });
    latex += "\\bottomrule\n\\end{tabular}\n\\caption{Ablation study on " + selectedDs + "}\\end{table}";
    navigator.clipboard.writeText(latex);
    alert("LaTeX table copied to clipboard!");
  };

  return (
    <section id="benchmarks" className="py-20 border-t border-border/80 bg-surface/20 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 gap-4">
          <div>
            <p className="text-xs font-mono uppercase text-lime tracking-wider mb-2">
              // 04 EMPIRICAL BENCHMARKS & ABLATIONS_
            </p>
            <h2 className="font-display text-3xl sm:text-4xl font-extrabold text-text tracking-tight">
              Ablation Study: Component Impact
            </h2>
            <p className="text-muted mt-2 text-sm sm:text-base max-w-2xl">
              Empirical performance audit across enterprise credit card, underwriting, and sensor streams. Demonstrates strict sub-1% accuracy preservation with 13.4x footprint reduction and sub-50µs latency.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={exportCSV}
              className="px-3 py-1.5 rounded-lg bg-surface-2 border border-border hover:border-lime text-xs font-mono text-text flex items-center space-x-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5 text-lime" />
              <span>Export CSV</span>
            </button>
            <button
              onClick={exportLaTeX}
              className="px-3 py-1.5 rounded-lg bg-surface-2 border border-border hover:border-violet text-xs font-mono text-text flex items-center space-x-1.5 transition-colors"
            >
              <FileText className="w-3.5 h-3.5 text-violet" />
              <span>Copy LaTeX</span>
            </button>
          </div>
        </div>

        {/* Dataset Tabs */}
        <div className="flex items-center space-x-2 mb-6 border-b border-border/60 pb-3">
          <button
            onClick={() => setSelectedDs("credit_card_fraud")}
            className={`px-4 py-2 rounded-lg font-mono text-xs transition-colors ${
              selectedDs === "credit_card_fraud"
                ? "bg-surface-2 border border-lime text-lime font-bold"
                : "text-muted hover:text-text"
            }`}
          >
            Credit Card Fraud (62.6:1 Imbalance)
          </button>
          <button
            onClick={() => setSelectedDs("home_credit")}
            className={`px-4 py-2 rounded-lg font-mono text-xs transition-colors ${
              selectedDs === "home_credit"
                ? "bg-surface-2 border border-lime text-lime font-bold"
                : "text-muted hover:text-text"
            }`}
          >
            Home Credit Default Risk
          </button>
        </div>

        {/* Table Container */}
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-surface-2/90 border-b border-border text-muted uppercase text-[10px]">
                <tr>
                  <th className="py-3.5 px-4">Compression Method</th>
                  <th className="py-3.5 px-4">Model Size</th>
                  <th className="py-3.5 px-4">Edge Latency</th>
                  <th className="py-3.5 px-4">ROC-AUC</th>
                  <th className="py-3.5 px-4">PR-AUC</th>
                  <th className="py-3.5 px-4">Retained %</th>
                  <th className="py-3.5 px-4 text-right">Within 1-2% Gate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60">
                {rows.map((row, idx) => {
                  const isOurs = row.method.includes("TinyAI");
                  return (
                    <tr
                      key={idx}
                      className={
                        isOurs
                          ? "bg-lime/10 font-bold"
                          : "hover:bg-surface-2/40 transition-colors"
                      }
                    >
                      <td className="py-3.5 px-4 flex items-center space-x-2">
                        {isOurs && <Sparkles className="w-3.5 h-3.5 text-lime" />}
                        <span className={isOurs ? "text-lime" : "text-text"}>
                          {row.method}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-text">{row.size_kb} KB</td>
                      <td className="py-3.5 px-4 text-violet">{row.latency_ms} ms</td>
                      <td className="py-3.5 px-4 text-text">{row.roc_auc}</td>
                      <td className="py-3.5 px-4 text-text">{row.pr_auc}</td>
                      <td className="py-3.5 px-4 text-text">{row.retention_pct}%</td>
                      <td className="py-3.5 px-4 text-right">
                        <span
                          className={`inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                            row.within_tolerance ? "badge-pass" : "badge-fail"
                          }`}
                        >
                          {row.within_tolerance ? (
                            <>
                              <CheckCircle2 className="w-3 h-3 text-lime" />
                              <span>PASS</span>
                            </>
                          ) : (
                            <>
                              <XCircle className="w-3 h-3 text-magenta" />
                              <span>FAIL</span>
                            </>
                          )}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
