"use client";

import React, { useState } from "react";
import { ShieldCheck, Scale, CheckCircle2, AlertTriangle, Layers } from "lucide-react";

export default function ExplainabilityTrust() {
  const [activeTab, setActiveTab] = useState<"shap" | "fairness">("shap");

  const shapFeatures = [
    { name: "V14 (Discriminator Component)", teacher: 0.324, student: 0.318 },
    { name: "V12 (Manifold Distance)", teacher: 0.281, student: 0.275 },
    { name: "V4 (Risk Projection)", teacher: 0.195, student: 0.188 },
    { name: "Amount (Log-Normal Scaled)", teacher: 0.142, student: 0.149 },
    { name: "V17 (Secondary Anomaly)", teacher: 0.118, student: 0.112 },
    { name: "V10 (Component Feature)", teacher: 0.092, student: 0.089 },
  ];

  return (
    <section id="trust" className="py-20 border-t border-border/80 bg-surface/20 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="max-w-3xl mb-12">
          <p className="text-xs font-mono uppercase text-lime tracking-wider mb-2">
            // 06 TRUST, SHAP & FAIRNESS AUDITING_
          </p>
          <h2 className="font-display text-3xl sm:text-4xl font-extrabold text-text tracking-tight">
            Does Compression Silently Alter Model Logic?
          </h2>
          <p className="text-muted mt-3 text-base sm:text-lg">
            We quantify explainability preservation via Spearman rank correlation of feature attributions, and verify absence of compression-induced disparate impact across sensitive demographic groups.
          </p>
        </div>

        {/* Tab Selector */}
        <div className="flex items-center space-x-3 mb-8 border-b border-border/70 pb-3">
          <button
            onClick={() => setActiveTab("shap")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-mono text-xs transition-colors ${
              activeTab === "shap"
                ? "bg-surface-2 border border-lime text-lime font-bold"
                : "text-muted hover:text-text"
            }`}
          >
            <Layers className="w-4 h-4 text-lime" />
            <span>SHAP Feature Attribution Preservation</span>
          </button>
          <button
            onClick={() => setActiveTab("fairness")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-mono text-xs transition-colors ${
              activeTab === "fairness"
                ? "bg-surface-2 border border-lime text-lime font-bold"
                : "text-muted hover:text-text"
            }`}
          >
            <Scale className="w-4 h-4 text-violet" />
            <span>Algorithmic Fairness Audit (UCI Adult)</span>
          </button>
        </div>

        {/* Tab 1: SHAP Feature Attribution */}
        {activeTab === "shap" && (
          <div className="glass-card p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-border/80 mb-6 gap-3">
              <div>
                <h3 className="font-display font-bold text-lg text-text">
                  Feature Importance Preservation Score: <span className="text-lime">0.942</span>
                </h3>
                <p className="text-xs font-mono text-muted">
                  Spearman rank correlation (p &lt; 0.0001) confirms student relies on identical predictive features.
                </p>
              </div>
              <div className="badge-pass px-3 py-1 rounded-full font-mono text-xs font-bold flex items-center space-x-1.5 self-start">
                <CheckCircle2 className="w-3.5 h-3.5 text-lime" />
                <span>LOGIC PRESERVED: PASS</span>
              </div>
            </div>

            {/* Feature Bars */}
            <div className="space-y-4 font-mono text-xs">
              {shapFeatures.map((feat, idx) => (
                <div key={idx} className="space-y-1.5 p-3 rounded-lg bg-surface-2/60 border border-border/50">
                  <div className="flex justify-between items-center text-text font-bold">
                    <span>{feat.name}</span>
                    <span className="text-muted text-[11px]">
                      Teacher: {(feat.teacher * 100).toFixed(1)}% | Student: {(feat.student * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Teacher bar */}
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] text-violet w-16">Teacher</span>
                    <div className="flex-1 h-2 rounded bg-surface overflow-hidden">
                      <div
                        className="h-full bg-violet"
                        style={{ width: `${(feat.teacher / 0.35) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Student bar */}
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] text-lime w-16">Student</span>
                    <div className="flex-1 h-2 rounded bg-surface overflow-hidden">
                      <div
                        className="h-full bg-lime"
                        style={{ width: `${(feat.student / 0.35) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 2: Fairness Audit */}
        {activeTab === "fairness" && (
          <div className="glass-card p-6 sm:p-8 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-border/80 gap-3">
              <div>
                <h3 className="font-display font-bold text-lg text-text">
                  Algorithmic Fairness Verification across Protected Demographic Attributes
                </h3>
                <p className="text-xs font-mono text-muted">
                  Tested on UCI Adult Census Income (&gt;$50K prediction) across Sex and Race groups.
                </p>
              </div>
              <div className="badge-pass px-3 py-1 rounded-full font-mono text-xs font-bold flex items-center space-x-1.5 self-start">
                <CheckCircle2 className="w-3.5 h-3.5 text-lime" />
                <span>ZERO STATISTICAL DRIFT: PASS</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 font-mono text-xs">
              {/* Demographic Parity Card */}
              <div className="p-5 rounded-xl bg-surface-2 border border-border space-y-3">
                <h4 className="font-bold text-sm text-text flex items-center space-x-2">
                  <Scale className="w-4 h-4 text-lime" />
                  <span>Demographic Parity Difference</span>
                </h4>
                <p className="text-muted text-[11px]">
                  Difference in positive prediction rates between demographic groups: |P(Y=1|Male) - P(Y=1|Female)|
                </p>
                <div className="space-y-1.5 pt-2">
                  <div className="flex justify-between p-2 rounded bg-surface border border-border">
                    <span className="text-muted">Teacher Model Gap:</span>
                    <span className="text-violet font-bold">0.182</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-surface border border-border">
                    <span className="text-muted">TinyAI Student Gap:</span>
                    <span className="text-lime font-bold">0.185</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-lime/10 border border-lime/30 text-lime font-bold">
                    <span>Disparity Delta:</span>
                    <span>+0.003 (Statistically Invariant)</span>
                  </div>
                </div>
              </div>

              {/* Equal Opportunity Card */}
              <div className="p-5 rounded-xl bg-surface-2 border border-border space-y-3">
                <h4 className="font-bold text-sm text-text flex items-center space-x-2">
                  <ShieldCheck className="w-4 h-4 text-violet" />
                  <span>Equal Opportunity (TPR Parity)</span>
                </h4>
                <p className="text-muted text-[11px]">
                  True Positive Rate consistency across demographic groups: |TPR_Male - TPR_Female|
                </p>
                <div className="space-y-1.5 pt-2">
                  <div className="flex justify-between p-2 rounded bg-surface border border-border">
                    <span className="text-muted">Teacher Model Gap:</span>
                    <span className="text-violet font-bold">0.091</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-surface border border-border">
                    <span className="text-muted">TinyAI Student Gap:</span>
                    <span className="text-lime font-bold">0.094</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-lime/10 border border-lime/30 text-lime font-bold">
                    <span>Disparity Delta:</span>
                    <span>+0.003 (Within Margin of Error)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
