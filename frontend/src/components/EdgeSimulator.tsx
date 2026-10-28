"use client";

import React, { useState } from "react";
import { Cpu, HardDrive, Battery, Gauge, CheckCircle2, XCircle, AlertCircle } from "lucide-react";

const DEVICES = [
  { id: "rpi4", name: "Raspberry Pi 4B (Cortex-A72)", ramMb: 4096, maxModelMb: 50.0, latencyBudgetMs: 15.0, powerW: 5.0, cpuFactor: 2.8 },
  { id: "jetson_nano", name: "NVIDIA Jetson Nano (Maxwell GPU)", ramMb: 2048, maxModelMb: 40.0, latencyBudgetMs: 10.0, powerW: 10.0, cpuFactor: 2.1 },
  { id: "cortex_m", name: "ARM Cortex-M55 Microcontroller", ramMb: 1.0, maxModelMb: 0.35, latencyBudgetMs: 50.0, powerW: 0.25, cpuFactor: 18.0 },
  { id: "pos_terminal", name: "Enterprise POS Terminal (Verifone)", ramMb: 512, maxModelMb: 15.0, latencyBudgetMs: 25.0, powerW: 3.0, cpuFactor: 4.5 },
  { id: "android_edge", name: "Handheld Android Device (Helio P22)", ramMb: 1536, maxModelMb: 25.0, latencyBudgetMs: 20.0, powerW: 2.5, cpuFactor: 3.2 },
];

export default function EdgeSimulator() {
  const [selectedDevice, setSelectedDevice] = useState(DEVICES[2]); // Cortex-M default
  const [selectedModel, setSelectedModel] = useState<"teacher" | "student">("student");

  // Model specs
  const modelSpecs = {
    teacher: { name: "Full Teacher Model (FP32)", sizeKb: 96.5, baseLatencyMs: 1.85 },
    student: { name: "TinyAI INT8 Student (Ours)", sizeKb: 7.2, baseLatencyMs: 0.045 },
  }[selectedModel];

  const modelRamMb = modelSpecs.sizeKb / 1024.0;
  const ramPassed = modelRamMb <= selectedDevice.maxModelMb;
  const ramUtilization = Math.min((modelRamMb / selectedDevice.maxModelMb) * 100, 100);

  const edgeLatencyMs = modelSpecs.baseLatencyMs * selectedDevice.cpuFactor;
  const latencyPassed = edgeLatencyMs <= selectedDevice.latencyBudgetMs;
  const latencyUtilization = Math.min((edgeLatencyMs / selectedDevice.latencyBudgetMs) * 100, 100);

  const overallPassed = ramPassed && latencyPassed;

  return (
    <section id="edge-sim" className="py-20 border-t border-border/80 bg-surface/30 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="max-w-3xl mb-12">
          <p className="text-xs font-mono uppercase text-violet tracking-wider mb-2">
            // 05 EDGE HARDWARE SIMULATOR_
          </p>
          <h2 className="font-display text-3xl sm:text-4xl font-extrabold text-text tracking-tight">
            Will It Run On Your Edge Target?
          </h2>
          <p className="text-muted mt-3 text-base sm:text-lg">
            Verify whether model variants fit tight SRAM ceilings and strict latency budgets across embedded processors, microcontrollers, and point-of-sale systems.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left: Device Selection & Model Variant Toggle */}
          <div className="lg:col-span-5 space-y-6">
            <div className="glass-card p-6 space-y-6">
              <div>
                <label className="text-xs font-mono uppercase text-muted block mb-2.5">
                  // SELECT TARGET HARDWARE_
                </label>
                <div className="space-y-2">
                  {DEVICES.map((d) => {
                    const isSelected = selectedDevice.id === d.id;
                    return (
                      <button
                        key={d.id}
                        onClick={() => setSelectedDevice(d)}
                        className={`w-full p-3 rounded-lg border text-left font-mono text-xs transition-all flex items-center justify-between ${
                          isSelected
                            ? "bg-surface-2 border-lime text-text shadow-lime-glow font-bold"
                            : "bg-surface border-border text-muted hover:border-violet"
                        }`}
                      >
                        <div>
                          <p>{d.name}</p>
                          <span className="text-[10px] text-muted">
                            RAM: {d.ramMb} MB | Budget: {d.latencyBudgetMs} ms | {d.powerW}W
                          </span>
                        </div>
                        {isSelected && <span className="text-lime text-xs">SELECTED</span>}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Model Variant Selector */}
              <div className="pt-2 border-t border-border/60">
                <label className="text-xs font-mono uppercase text-muted block mb-2">
                  // EVALUATE MODEL VARIANT_
                </label>
                <div className="grid grid-cols-2 gap-2 font-mono text-xs">
                  <button
                    onClick={() => setSelectedModel("teacher")}
                    className={`py-2 px-3 rounded border text-left transition-colors ${
                      selectedModel === "teacher"
                        ? "bg-violet/20 border-violet text-violet font-bold"
                        : "bg-surface border-border text-muted"
                    }`}
                  >
                    Teacher (96.5 KB)
                  </button>
                  <button
                    onClick={() => setSelectedModel("student")}
                    className={`py-2 px-3 rounded border text-left transition-colors ${
                      selectedModel === "student"
                        ? "bg-lime/20 border-lime text-lime font-bold"
                        : "bg-surface border-border text-muted"
                    }`}
                  >
                    TinyAI Student (7.2 KB)
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Hardware Gauges & Pass/Fail Output */}
          <div className="lg:col-span-7 glass-card p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-border/80 mb-6">
                <div>
                  <h3 className="font-display font-extrabold text-lg text-text">
                    Hardware Fit Analysis
                  </h3>
                  <p className="text-xs font-mono text-muted">
                    {selectedDevice.name}
                  </p>
                </div>

                <div
                  className={`px-3 py-1 rounded-full font-mono text-xs font-bold flex items-center space-x-1.5 ${
                    overallPassed ? "badge-pass" : "badge-fail"
                  }`}
                >
                  {overallPassed ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 text-lime" />
                      <span>DEPLOYABLE: PASS</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="w-4 h-4 text-magenta" />
                      <span>BUDGET EXCEEDED: FAIL</span>
                    </>
                  )}
                </div>
              </div>

              {/* Resource Gauges */}
              <div className="space-y-6 font-mono text-xs">
                {/* RAM Gauge */}
                <div className="p-4 rounded-xl bg-surface-2 border border-border space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center space-x-2 text-muted">
                      <HardDrive className="w-4 h-4 text-violet" />
                      <span>SRAM / RAM Ceiling</span>
                    </span>
                    <span className={ramPassed ? "text-lime font-bold" : "text-magenta font-bold"}>
                      {(modelRamMb * 1024).toFixed(1)} KB / {(selectedDevice.maxModelMb * 1024).toFixed(0)} KB ({ramUtilization.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full h-3 rounded-full bg-surface overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        ramPassed ? "bg-lime" : "bg-magenta"
                      }`}
                      style={{ width: `${ramUtilization}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-muted">
                    {ramPassed
                      ? "✓ Model fits comfortably within allocated edge SRAM."
                      : "✕ Model exceeds physical edge memory limit! Device will suffer Out-Of-Memory panic."}
                  </p>
                </div>

                {/* Latency Gauge */}
                <div className="p-4 rounded-xl bg-surface-2 border border-border space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center space-x-2 text-muted">
                      <Gauge className="w-4 h-4 text-lime" />
                      <span>Execution Latency Budget</span>
                    </span>
                    <span className={latencyPassed ? "text-lime font-bold" : "text-magenta font-bold"}>
                      {edgeLatencyMs.toFixed(3)} ms / {selectedDevice.latencyBudgetMs} ms ({latencyUtilization.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full h-3 rounded-full bg-surface overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        latencyPassed ? "bg-lime" : "bg-magenta"
                      }`}
                      style={{ width: `${latencyUtilization}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-muted">
                    {latencyPassed
                      ? "✓ Satisfies point-of-sale transaction timeout constraint."
                      : "✕ Inference execution too slow for time-sensitive edge interaction."}
                  </p>
                </div>
              </div>
            </div>

            {/* Power Estimate Footer */}
            <div className="mt-6 pt-4 border-t border-border/80 flex items-center justify-between font-mono text-xs text-muted">
              <div className="flex items-center space-x-2">
                <Battery className="w-4 h-4 text-amber" />
                <span>Estimated Energy Consumption:</span>
              </div>
              <span className="text-amber font-bold">
                {selectedModel === "student" ? "0.004 mJ / inference" : "0.089 mJ / inference (22x more)"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
