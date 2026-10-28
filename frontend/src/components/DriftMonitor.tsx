"use client";

import React, { useState } from "react";
import { Activity, AlertTriangle, CheckCircle2, RefreshCw, Zap } from "lucide-react";
import { simulateDrift } from "@/lib/api";

export default function DriftMonitor() {
  const [driftSeverity, setDriftSeverity] = useState(0.25);
  const [isSimulating, setIsSimulating] = useState(false);
  const [driftReport, setDriftReport] = useState<any>({
    max_psi: 0.18,
    overall_status: "MODERATE DRIFT",
    recompression_required: false,
    recommendation: "Edge pipeline within acceptable operating tolerances.",
    features: [
      { feature: "Time_Interval", psi: 0.18, ks_statistic: 0.14, p_value: 0.021, status: "MODERATE" },
      { feature: "Transaction_Amount", psi: 0.12, ks_statistic: 0.09, p_value: 0.084, status: "MODERATE" },
      { feature: "V14_Component", psi: 0.04, ks_statistic: 0.03, p_value: 0.52, status: "STABLE" },
      { feature: "V4_Component", psi: 0.03, ks_statistic: 0.02, p_value: 0.74, status: "STABLE" },
    ],
  });

  const handleSimulate = async () => {
    setIsSimulating(true);
    const report = await simulateDrift("credit_card_fraud", driftSeverity);
    setDriftReport(report);
    setIsSimulating(false);
  };

  return (
    <section id="drift" className="py-20 border-t border-border/80 bg-surface/30 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="max-w-3xl mb-12">
          <p className="text-xs font-mono uppercase text-amber tracking-wider mb-2">
            // 07 PRODUCTION DRIFT MONITORING_
          </p>
          <h2 className="font-display text-3xl sm:text-4xl font-extrabold text-text tracking-tight">
            Edge Covariate Drift & Re-Scheduling
          </h2>
          <p className="text-muted mt-3 text-base sm:text-lg">
            Monitor deployed edge streams in real time. When Population Stability Index (PSI) exceeds 0.25, TinyAI automatically alerts engineers and triggers automated re-compression.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left: Drift Injection Slider */}
          <div className="lg:col-span-5 glass-card p-6 space-y-6">
            <h3 className="font-display font-bold text-base text-text flex items-center space-x-2">
              <Activity className="w-4 h-4 text-amber" />
              <span>Simulate Edge Stream Shift</span>
            </h3>

            <div className="space-y-4 font-mono text-xs">
              <div>
                <div className="flex justify-between mb-1.5">
                  <span className="text-muted">Covariate Drift Intensity:</span>
                  <span className="text-amber font-bold">{(driftSeverity * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.80"
                  step="0.05"
                  value={driftSeverity}
                  onChange={(e) => setDriftSeverity(Number(e.target.value))}
                  className="w-full accent-amber bg-surface-2 cursor-pointer h-1.5 rounded"
                />
                <span className="text-[10px] text-muted">
                  Simulates seasonal changes or evolving fraud attack patterns.
                </span>
              </div>

              <button
                disabled={isSimulating}
                onClick={handleSimulate}
                className="w-full btn-primary py-2.5 text-xs flex items-center justify-center space-x-2"
              >
                {isSimulating ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Zap className="w-3.5 h-3.5 text-black" />
                )}
                <span>Inject Shift & Run PSI / KS-Test</span>
              </button>

              <div className="p-3.5 rounded-lg bg-surface-2/80 border border-border text-[11px] text-muted space-y-1">
                <span className="block font-bold text-text uppercase text-[10px]">// PSI THRESHOLD RULES:</span>
                <p>• PSI &lt; 0.10: Stable (Green)</p>
                <p>• 0.10 ≤ PSI ≤ 0.25: Moderate Drift (Amber)</p>
                <p>• PSI &gt; 0.25: Critical Drift (Magenta) → Recompression Required</p>
              </div>
            </div>
          </div>

          {/* Right: Drift Test Results Table */}
          <div className="lg:col-span-7 glass-card p-6 space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-border/80">
              <div>
                <h4 className="font-display font-bold text-base text-text">
                  Multi-Feature Distribution Shift Report
                </h4>
                <p className="text-[11px] font-mono text-muted">
                  Max PSI: <span className="text-text font-bold">{driftReport.max_psi}</span>
                </p>
              </div>
              <span
                className={`px-3 py-1 rounded-full font-mono text-xs font-bold ${
                  driftReport.recompression_required ? "badge-fail" : "badge-pass"
                }`}
              >
                {driftReport.overall_status}
              </span>
            </div>

            {/* Feature Drift Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-surface-2 text-muted uppercase text-[10px]">
                  <tr>
                    <th className="py-2.5 px-3">Feature Name</th>
                    <th className="py-2.5 px-3">PSI Index</th>
                    <th className="py-2.5 px-3">KS Statistic</th>
                    <th className="py-2.5 px-3">p-value</th>
                    <th className="py-2.5 px-3 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60">
                  {driftReport.features.map((f: any, idx: number) => (
                    <tr key={idx} className="hover:bg-surface-2/30">
                      <td className="py-2.5 px-3 text-text font-bold">{f.feature}</td>
                      <td className="py-2.5 px-3 text-amber">{f.psi.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-muted">{f.ks_statistic.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-muted">{f.p_value.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-right">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            f.status === "CRITICAL"
                              ? "bg-magenta/20 text-magenta"
                              : f.status === "MODERATE"
                              ? "bg-amber/20 text-amber"
                              : "bg-lime/20 text-lime"
                          }`}
                        >
                          {f.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Recompression Trigger Action */}
            {driftReport.recompression_required && (
              <div className="p-4 rounded-xl bg-magenta/15 border border-magenta/40 flex items-center justify-between font-mono text-xs">
                <div className="flex items-center space-x-2 text-magenta font-bold">
                  <AlertTriangle className="w-5 h-5 text-magenta animate-pulse" />
                  <span>Critical shift detected! Decision manifold requires adaptation.</span>
                </div>
                <a
                  href="#scheduler"
                  className="px-3 py-1.5 rounded bg-magenta text-black font-bold hover:bg-magenta/90 transition-colors"
                >
                  Trigger Recompression
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
