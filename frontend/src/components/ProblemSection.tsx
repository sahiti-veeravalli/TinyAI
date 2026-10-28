"use client";

import React from "react";
import { Server, Smartphone, BatteryCharging, Zap, Shield, Clock, HardDrive, AlertTriangle } from "lucide-react";

export default function ProblemSection() {
  const comparisons = [
    {
      title: "Memory Footprint",
      cloud: "Large Ensembles / MLPs: 50MB – 500MB RAM required",
      edge: "Cortex-M / POS: Under 1MB SRAM available",
      solution: "TinyAI 13.4x compression: 7.2 KB ONNX runtime footprint",
      icon: HardDrive,
    },
    {
      title: "Latency & Round-Trip",
      cloud: "Network ping + Cloud queue: 150ms – 400ms delay",
      edge: "Point-of-sale timeout budget: <25ms strictly enforced",
      solution: "TinyAI On-Device: 0.045ms – 1.15ms local inference",
      icon: Clock,
    },
    {
      title: "Network & Offline Privacy",
      cloud: "Transmits raw applicant financial/PII data over WAN",
      edge: "Branch devices intermittently offline; strict GDPR / PCI-DSS",
      solution: "Fully air-gapped on-chip inference with zero data leakage",
      icon: Shield,
    },
    {
      title: "Power & Thermal Envelope",
      cloud: "Server racks consuming >350W per compute node",
      edge: "Battery & solar IoT gateways: <2.5W thermal dissipation",
      solution: "Zero active GPU needed; runs on low-power integer ALU",
      icon: BatteryCharging,
    },
  ];

  return (
    <section id="overview" className="py-20 border-t border-border/80 bg-surface/40 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="max-w-3xl mb-14">
          <p className="text-xs font-mono uppercase text-lime tracking-wider mb-2">
            // 01 THE PROBLEM_
          </p>
          <h2 className="font-display text-3xl sm:text-4xl font-extrabold text-text tracking-tight">
            Enterprise analytics cannot wait for the cloud.
          </h2>
          <p className="text-muted mt-3 text-base sm:text-lg">
            High-accuracy fraud detection, credit underwriting, and churn prediction models are too memory-heavy for edge hardware, while naive compression destroys recall on rare anomalies.
          </p>
        </div>

        {/* Comparison Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {comparisons.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div key={idx} className="glass-card p-6 relative group overflow-hidden">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="p-2 rounded-lg bg-surface-2 border border-border text-lime group-hover:border-lime transition-colors">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h3 className="font-display font-bold text-lg text-text">{item.title}</h3>
                </div>

                <div className="space-y-3 font-mono text-xs">
                  <div className="p-3 rounded bg-surface-2/80 border border-border/60">
                    <span className="text-muted block text-[10px] uppercase">// CLOUD BOTTLENECK</span>
                    <span className="text-magenta">{item.cloud}</span>
                  </div>

                  <div className="p-3 rounded bg-surface-2/80 border border-border/60">
                    <span className="text-muted block text-[10px] uppercase">// EDGE CONSTRAINT</span>
                    <span className="text-amber">{item.edge}</span>
                  </div>

                  <div className="p-3 rounded bg-lime/10 border border-lime/30">
                    <span className="text-lime block text-[10px] uppercase font-bold">// TINYAI SOLUTION</span>
                    <span className="text-text font-medium">{item.solution}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
