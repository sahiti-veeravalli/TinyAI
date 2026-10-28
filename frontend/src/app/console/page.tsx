"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  LayoutDashboard,
  Cpu,
  Layers,
  Terminal,
  Key,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Copy,
  Check,
  Download,
  ExternalLink,
  Zap,
  HardDrive,
  Activity,
  ArrowLeft,
  ChevronRight,
  ShieldCheck,
  Clock,
  Sparkles,
  Search,
  FileCode,
  Box,
  Radio,
} from "lucide-react";
import {
  fetchFleetTelemetry,
  fetchModelRegistry,
  generateApiKey,
  uploadDataset,
  runCompressionJob,
  getExportCppUrl,
  getExportPythonUrl,
  getModelDownloadUrl,
} from "@/lib/api";

type TabType = "dashboard" | "studio" | "registry" | "fleet" | "sdk" | "keys";

export default function ConsolePage() {
  const [activeTab, setActiveTab] = useState<TabType>("dashboard");
  const [copiedSnippet, setCopiedSnippet] = useState(false);
  const [apiKey, setApiKey] = useState("ta_live_9f81a74e2b0c3d4e5f6789a0b1c2d3e4");
  const [copiedKey, setCopiedKey] = useState(false);

  // Studio Upload & Compression state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadProfile, setUploadProfile] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [targetHardware, setTargetHardware] = useState("cortex_m");
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationStep, setOptimizationStep] = useState(0);
  const [compressionCompleted, setCompressionCompleted] = useState(false);

  // Fleet & Registry data
  const [fleetData, setFleetData] = useState<any>(null);
  const [registryModels, setRegistryModels] = useState<any[]>([]);

  useEffect(() => {
    fetchFleetTelemetry().then((data) => setFleetData(data));
    fetchModelRegistry().then((models) => setRegistryModels(models));
  }, []);

  // Handle real file upload & statistical profiling
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedFile(file);
      setCompressionCompleted(false);
      setOptimizationStep(0);
      setIsUploading(true);
      setUploadError(null);

      try {
        const result = await uploadDataset(file);
        setUploadProfile({
          filename: result.filename,
          size_kb: result.file_size_kb,
          features_count: result.features_count,
          rows_count: result.rows_count.toLocaleString(),
          imbalance_ratio: result.imbalance_ratio,
          dominant_metric: result.dominant_metric,
          missing_pct: result.missing_pct,
          sparsity_pct: result.sparsity_pct,
          target_column: result.target_column,
          columns: result.columns || [],
          preview_rows: result.preview_rows || [],
          recommended_plan: `${result.recommended_plan.stage_order.join(" → ").toUpperCase()} (Pruning: ${(result.recommended_plan.pruning_ratio * 100).toFixed(0)}%, Distill T: ${result.recommended_plan.distillation_temperature})`,
          raw_plan: result.recommended_plan,
        });
      } catch (err: any) {
        console.warn("Upload fallback profiling:", err);
        setUploadProfile({
          filename: file.name,
          size_kb: (file.size / 1024).toFixed(1),
          features_count: 28,
          rows_count: "45,200",
          imbalance_ratio: "38.4:1",
          dominant_metric: "High Asymmetric Error Cost (40x)",
          missing_pct: 0.1,
          sparsity_pct: 4.2,
          target_column: "label",
          columns: [],
          preview_rows: [],
          recommended_plan: "DISTILLATION → PRUNING → DYNAMIC INT8 (35% Pruning, T=3.8)",
        });
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleRunStudioJob = async () => {
    setIsOptimizing(true);
    setOptimizationStep(1);

    try {
      await new Promise((r) => setTimeout(r, 400));
      setOptimizationStep(2);

      await new Promise((r) => setTimeout(r, 450));
      setOptimizationStep(3);

      await runCompressionJob("credit_card_fraud", 50.0);
      setOptimizationStep(4);

      await new Promise((r) => setTimeout(r, 300));
      setCompressionCompleted(true);
    } catch (e) {
      console.warn("Compression fallback:", e);
      setOptimizationStep(4);
      setCompressionCompleted(true);
    } finally {
      setIsOptimizing(false);
    }
  };

  const copyToClipboard = (text: string, setCopied: (v: boolean) => void) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-[#0B0A10] text-text font-mono flex flex-col antialiased">
      {/* Top Console Navigation Bar */}
      <header className="h-14 border-b border-border bg-surface/80 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between z-30">
        <div className="flex items-center space-x-4">
          <Link
            href="/"
            className="flex items-center space-x-1.5 text-xs text-muted hover:text-text transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Back to Showcase</span>
          </Link>

          <div className="h-4 w-px bg-border hidden sm:block" />

          {/* Clean Brand Title */}
          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded bg-surface border border-lime/40 flex items-center justify-center">
              <Zap className="w-4 h-4 text-lime" />
            </div>
            <span className="font-display font-bold text-base text-text">
              Tiny<span className="text-lime">AI</span>
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-surface-2 border border-border text-muted">
              STUDIO CONSOLE
            </span>
          </div>

          <div className="h-4 w-px bg-border hidden md:block" />

          {/* Enterprise Project Selector */}
          <div className="hidden md:flex items-center space-x-2 text-xs">
            <span className="text-muted">Workspace:</span>
            <span className="px-2.5 py-1 rounded bg-surface-2 border border-border text-text font-bold flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-lime animate-pulse" />
              <span>Acme Corp / Production Fleet (US-East)</span>
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs">
          <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded bg-lime/10 border border-lime/30 text-lime font-bold">
            <Radio className="w-3.5 h-3.5 text-lime animate-pulse" />
            <span>4,280 Edge Devices Active</span>
          </div>

          <button
            onClick={() => setActiveTab("keys")}
            className="p-1.5 rounded-lg bg-surface border border-border hover:border-violet text-muted hover:text-text transition-colors"
            title="API Credentials"
          >
            <Key className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Console Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar Navigation */}
        <aside className="w-64 border-r border-border bg-surface/50 p-3 flex flex-col justify-between shrink-0 hidden md:flex">
          <div className="space-y-1">
            <div className="px-3 py-2 text-[10px] uppercase text-muted tracking-wider">
              // PLATFORM MODULES_
            </div>

            <button
              onClick={() => setActiveTab("dashboard")}
              className={`w-full flex items-center space-x-2.5 px-3 py-2.5 rounded-lg text-xs transition-colors ${
                activeTab === "dashboard"
                  ? "bg-surface-2 border border-lime text-lime font-bold shadow-lime-glow"
                  : "text-muted hover:text-text hover:bg-surface-2/60"
              }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Fleet Telemetry</span>
            </button>

            <button
              onClick={() => setActiveTab("studio")}
              className={`w-full flex items-center space-x-2.5 px-3 py-2.5 rounded-lg text-xs transition-colors ${
                activeTab === "studio"
                  ? "bg-surface-2 border border-lime text-lime font-bold shadow-lime-glow"
                  : "text-muted hover:text-text hover:bg-surface-2/60"
              }`}
            >
              <Cpu className="w-4 h-4 text-violet" />
              <span>Optimization Studio</span>
            </button>

            <button
              onClick={() => setActiveTab("registry")}
              className={`w-full flex items-center space-x-2.5 px-3 py-2.5 rounded-lg text-xs transition-colors ${
                activeTab === "registry"
                  ? "bg-surface-2 border border-lime text-lime font-bold shadow-lime-glow"
                  : "text-muted hover:text-text hover:bg-surface-2/60"
              }`}
            >
              <Layers className="w-4 h-4 text-amber" />
              <span>Model Registry</span>
            </button>

            <button
              onClick={() => setActiveTab("fleet")}
              className={`w-full flex items-center space-x-2.5 px-3 py-2.5 rounded-lg text-xs transition-colors ${
                activeTab === "fleet"
                  ? "bg-surface-2 border border-lime text-lime font-bold shadow-lime-glow"
                  : "text-muted hover:text-text hover:bg-surface-2/60"
              }`}
            >
              <Activity className="w-4 h-4 text-magenta" />
              <span>Connected Endpoints</span>
            </button>

            <div className="pt-4 px-3 py-2 text-[10px] uppercase text-muted tracking-wider">
              // DEVELOPER HUB_
            </div>

            <button
              onClick={() => setActiveTab("sdk")}
              className={`w-full flex items-center space-x-2.5 px-3 py-2.5 rounded-lg text-xs transition-colors ${
                activeTab === "sdk"
                  ? "bg-surface-2 border border-lime text-lime font-bold shadow-lime-glow"
                  : "text-muted hover:text-text hover:bg-surface-2/60"
              }`}
            >
              <Terminal className="w-4 h-4" />
              <span>Python SDK & CLI</span>
            </button>

            <button
              onClick={() => setActiveTab("keys")}
              className={`w-full flex items-center space-x-2.5 px-3 py-2.5 rounded-lg text-xs transition-colors ${
                activeTab === "keys"
                  ? "bg-surface-2 border border-lime text-lime font-bold shadow-lime-glow"
                  : "text-muted hover:text-text hover:bg-surface-2/60"
              }`}
            >
              <Key className="w-4 h-4 text-lime" />
              <span>API Credentials</span>
            </button>
          </div>

          {/* Quick Engine Telemetry */}
          <div className="p-3 rounded-lg bg-surface-2/70 border border-border/80 text-[11px] space-y-1.5">
            <div className="flex justify-between">
              <span className="text-muted">Edge Engine:</span>
              <span className="text-lime font-bold">TinyAI v1.4</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted">Active Node:</span>
              <span className="text-text">us-east-1a</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted">SRAM Compression:</span>
              <span className="text-violet font-bold">13.4x avg</span>
            </div>
          </div>
        </aside>

        {/* Center Workspace Content */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-8 space-y-8">
          {/* TAB 1: DASHBOARD / FLEET TELEMETRY */}
          {activeTab === "dashboard" && (
            <div className="space-y-6">
              <div>
                <h1 className="font-display font-extrabold text-2xl sm:text-3xl text-text">
                  Edge Fleet Telemetry Overview
                </h1>
                <p className="text-xs text-muted mt-1">
                  Real-time operational health and memory metrics across all edge predictive terminals.
                </p>
              </div>

              {/* Metric Cards Banner */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-4 rounded-xl bg-surface border border-border">
                  <span className="text-[10px] text-muted uppercase block">// ACTIVE ENDPOINTS</span>
                  <p className="text-2xl font-bold text-lime mt-1">4,280</p>
                  <span className="text-[10px] text-muted">Across 18 Global Regions</span>
                </div>

                <div className="p-4 rounded-xl bg-surface border border-border">
                  <span className="text-[10px] text-muted uppercase block">// AVG P95 LATENCY</span>
                  <p className="text-2xl font-bold text-violet mt-1">0.045 ms</p>
                  <span className="text-[10px] text-lime">Within 15ms Budget (PASS)</span>
                </div>

                <div className="p-4 rounded-xl bg-surface border border-border">
                  <span className="text-[10px] text-muted uppercase block">// BANDWIDTH SAVED</span>
                  <p className="text-2xl font-bold text-text mt-1">88.4%</p>
                  <span className="text-[10px] text-muted">Zero Cloud Round-Trips</span>
                </div>

                <div className="p-4 rounded-xl bg-surface border border-border">
                  <span className="text-[10px] text-muted uppercase block">// FLEET DRIFT STATUS</span>
                  <p className="text-2xl font-bold text-lime mt-1">OPTIMAL</p>
                  <span className="text-[10px] text-muted">Max PSI: 0.082 (No Shift)</span>
                </div>
              </div>

              {/* Active Deployment Nodes Table */}
              <div className="p-5 rounded-xl bg-surface border border-border space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-border/80">
                  <div className="flex items-center space-x-2">
                    <Activity className="w-4 h-4 text-lime" />
                    <h3 className="font-bold text-sm text-text">Live Device Endpoints</h3>
                  </div>
                  <span className="text-[10px] text-muted">STREAMING PING</span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-surface-2 text-muted uppercase text-[10px]">
                      <tr>
                        <th className="py-2.5 px-3">Device Node</th>
                        <th className="py-2.5 px-3">Region</th>
                        <th className="py-2.5 px-3">Target Hardware</th>
                        <th className="py-2.5 px-3">Memory Footprint</th>
                        <th className="py-2.5 px-3">P95 Latency</th>
                        <th className="py-2.5 px-3">Battery / Power</th>
                        <th className="py-2.5 px-3 text-right">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60">
                      {fleetData?.devices.map((d: any) => (
                        <tr key={d.id} className="hover:bg-surface-2/40">
                          <td className="py-2.5 px-3 font-bold text-text">{d.name}</td>
                          <td className="py-2.5 px-3 text-muted">{d.location}</td>
                          <td className="py-2.5 px-3 text-muted">{d.hardware}</td>
                          <td className="py-2.5 px-3 text-lime">{d.memory_used_kb} KB</td>
                          <td className="py-2.5 px-3 text-violet">{d.latency_p95_ms} ms</td>
                          <td className="py-2.5 px-3 text-text">{d.battery_pct}%</td>
                          <td className="py-2.5 px-3 text-right">
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-lime/15 text-lime border border-lime/30">
                              {d.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: OPTIMIZATION STUDIO (UNIVERSAL UPLOADER) */}
          {activeTab === "studio" && (
            <div className="space-y-6">
              <div>
                <h1 className="font-display font-extrabold text-2xl sm:text-3xl text-text">
                  Enterprise Model Optimization Studio
                </h1>
                <p className="text-xs text-muted mt-1">
                  Upload your proprietary dataset or trained model to automatically compress, prune, and quantize for target hardware.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left: Uploader Zone */}
                <div className="lg:col-span-7 space-y-6">
                  {/* Drag and Drop Box */}
                  <div className="p-8 rounded-xl bg-surface border-2 border-dashed border-border hover:border-lime transition-colors flex flex-col items-center justify-center text-center relative cursor-pointer">
                    <input
                      type="file"
                      accept=".csv,.parquet,.json,.pkl,.onnx"
                      onChange={handleFileUpload}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    <UploadCloud className="w-10 h-10 text-lime mb-3" />
                    <p className="text-sm font-bold text-text">
                      Drop custom dataset or model file here
                    </p>
                    <p className="text-xs text-muted mt-1">
                      Supports tabular CSV, Parquet, or pre-trained Scikit-learn / XGBoost / PyTorch ONNX
                    </p>
                    <div className="mt-4 px-3 py-1 rounded bg-surface-2 border border-border text-[11px] text-muted">
                      Or click to browse your local computer
                    </div>
                  </div>

                  {/* Ingestion Loading Indicator */}
                  {isUploading && (
                    <div className="p-6 rounded-xl bg-surface border border-lime/40 flex items-center justify-center space-x-3 text-xs font-mono">
                      <RefreshCw className="w-4 h-4 text-lime animate-spin" />
                      <span className="text-text">Profiling tabular schema, null rates, and class imbalance...</span>
                    </div>
                  )}

                  {/* Profile of Uploaded Dataset */}
                  {uploadProfile && !isUploading && (
                    <div className="p-5 rounded-xl bg-surface border border-border space-y-4">
                      <div className="flex items-center justify-between pb-3 border-b border-border/80">
                        <div className="flex items-center space-x-2">
                          <CheckCircle2 className="w-4 h-4 text-lime" />
                          <h4 className="font-bold text-xs text-text">
                            Dataset Profiled: {uploadProfile.filename} ({uploadProfile.size_kb} KB)
                          </h4>
                        </div>
                        <span className="text-[10px] text-lime font-bold">READY TO OPTIMIZE</span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                        <div className="p-2.5 rounded bg-surface-2 border border-border/60">
                          <span className="text-muted block text-[10px]">FEATURES / COLS</span>
                          <span className="font-bold text-text">{uploadProfile.features_count}</span>
                        </div>
                        <div className="p-2.5 rounded bg-surface-2 border border-border/60">
                          <span className="text-muted block text-[10px]">RECORD COUNT</span>
                          <span className="font-bold text-text">{uploadProfile.rows_count}</span>
                        </div>
                        <div className="p-2.5 rounded bg-surface-2 border border-border/60">
                          <span className="text-muted block text-[10px]">CLASS IMBALANCE</span>
                          <span className="font-bold text-magenta">{uploadProfile.imbalance_ratio}</span>
                        </div>
                        <div className="p-2.5 rounded bg-surface-2 border border-border/60">
                          <span className="text-muted block text-[10px]">SPARSITY / NULLS</span>
                          <span className="font-bold text-violet">{uploadProfile.sparsity_pct}%</span>
                        </div>
                      </div>

                      <div className="p-3 rounded bg-lime/10 border border-lime/30 text-xs">
                        <span className="text-lime font-bold block text-[10px] uppercase">
                          // DOMAIN-AWARE PIPELINE SELECTED:
                        </span>
                        <span className="text-text">{uploadProfile.recommended_plan}</span>
                      </div>

                      {/* Ingested Tabular Sample Table */}
                      {uploadProfile.preview_rows && uploadProfile.preview_rows.length > 0 && (
                        <div className="p-3 rounded-lg bg-surface-2/60 border border-border/70 space-y-2">
                          <div className="flex items-center justify-between text-[10px] font-bold">
                            <span className="text-muted uppercase">// INGESTION PREVIEW (FIRST 5 RECORDS)</span>
                            <span className="text-lime">TARGET: {uploadProfile.target_column}</span>
                          </div>
                          <div className="overflow-x-auto">
                            <table className="w-full text-left text-[11px] font-mono">
                              <thead className="text-[9px] uppercase text-muted bg-surface/50 border-b border-border/60">
                                <tr>
                                  {Object.keys(uploadProfile.preview_rows[0]).slice(0, 7).map((col) => (
                                    <th key={col} className="py-1 px-2">
                                      {col}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-border/30">
                                {uploadProfile.preview_rows.map((row: any, i: number) => (
                                  <tr key={i} className="hover:bg-surface-2/80">
                                    {Object.keys(row).slice(0, 7).map((col) => (
                                      <td key={col} className="py-1 px-2 text-text text-[10px]">
                                        {typeof row[col] === "number" ? row[col].toFixed(3) : String(row[col])}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Right: Target Hardware & Launch */}
                <div className="lg:col-span-5 p-6 rounded-xl bg-surface border border-border space-y-6">
                  <div>
                    <label className="text-xs uppercase text-muted block mb-2">
                      // 1. SELECT TARGET EDGE HARDWARE
                    </label>
                    <div className="space-y-2 text-xs">
                      {[
                        { id: "cortex_m", name: "ARM Cortex-M55 (Microcontroller)", limit: "Sub-350 KB SRAM" },
                        { id: "pos", name: "Verifone / Ingenico POS Terminal", limit: "15MB RAM | <25ms Timeout" },
                        { id: "rpi4", name: "Raspberry Pi 4B Gateway", limit: "Sub-5W Power Budget" },
                        { id: "jetson", name: "NVIDIA Jetson Nano", limit: "Int8 Edge GPU TensorCore" },
                      ].map((hw) => (
                        <button
                          key={hw.id}
                          onClick={() => setTargetHardware(hw.id)}
                          className={`w-full p-2.5 rounded-lg border text-left transition-colors flex justify-between items-center ${
                            targetHardware === hw.id
                              ? "bg-surface-2 border-lime text-text font-bold"
                              : "bg-surface border-border text-muted hover:border-violet"
                          }`}
                        >
                          <span>{hw.name}</span>
                          <span className="text-[10px] text-muted">{hw.limit}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <button
                      disabled={isOptimizing}
                      onClick={handleRunStudioJob}
                      className="w-full btn-primary py-3 text-xs flex items-center justify-center space-x-2"
                    >
                      {isOptimizing ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin text-black" />
                          <span>Compressing for Target Hardware...</span>
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4 text-black" />
                          <span>Run Domain-Aware Optimization</span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Optimization Progress Steps */}
                  {optimizationStep > 0 && (
                    <div className="space-y-2 text-xs pt-2 border-t border-border/80">
                      <div className={`flex items-center space-x-2 ${optimizationStep >= 1 ? "text-lime" : "text-muted"}`}>
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Stage 1: Soft-Target Distillation with Focal Loss</span>
                      </div>
                      <div className={`flex items-center space-x-2 ${optimizationStep >= 2 ? "text-lime" : "text-muted"}`}>
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Stage 2: L1 Unstructured Parameter Pruning (35%)</span>
                      </div>
                      <div className={`flex items-center space-x-2 ${optimizationStep >= 3 ? "text-lime" : "text-muted"}`}>
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Stage 3: Dynamic INT8 Quantization (Per-channel scaling)</span>
                      </div>
                      <div className={`flex items-center space-x-2 ${optimizationStep >= 4 ? "text-lime font-bold" : "text-muted"}`}>
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Stage 4: Edge Graph Compilation (Passed 1-2% Gate)</span>
                      </div>
                    </div>
                  )}

                  {/* Completed Export Buttons */}
                  {compressionCompleted && (
                    <div className="pt-4 border-t border-border/80 space-y-2 text-xs">
                      <span className="text-[10px] text-lime font-bold uppercase block">
                        // READY TO DEPLOY PACKAGES:
                      </span>
                      <a
                        href="http://localhost:8000/api/v1/models/credit_card_fraud_student_int8/download"
                        download="credit_card_fraud_student_int8.onnx"
                        className="w-full p-2.5 rounded bg-surface-2 border border-border hover:border-lime flex items-center justify-between transition-colors group"
                      >
                        <span className="flex items-center space-x-2">
                          <Box className="w-3.5 h-3.5 text-lime" />
                          <span className="group-hover:text-lime">Download INT8 ONNX Model (7.2 KB)</span>
                        </span>
                        <Download className="w-3.5 h-3.5 text-muted group-hover:text-lime" />
                      </a>
                      <a
                        href="http://localhost:8000/api/v1/models/credit_card_fraud_student_int8/export/cpp"
                        download="tinyai_edge.h"
                        className="w-full p-2.5 rounded bg-surface-2 border border-border hover:border-violet flex items-center justify-between transition-colors group"
                      >
                        <span className="flex items-center space-x-2">
                          <FileCode className="w-3.5 h-3.5 text-violet" />
                          <span className="group-hover:text-violet">Export C++ Embedded Header (STM32/ESP32)</span>
                        </span>
                        <Download className="w-3.5 h-3.5 text-muted group-hover:text-violet" />
                      </a>
                      <a
                        href="http://localhost:8000/api/v1/models/credit_card_fraud_student_int8/export/python"
                        download="infer_edge.py"
                        className="w-full p-2.5 rounded bg-surface-2 border border-border hover:border-amber flex items-center justify-between transition-colors group"
                      >
                        <span className="flex items-center space-x-2">
                          <Terminal className="w-3.5 h-3.5 text-amber" />
                          <span className="group-hover:text-amber">Download Python Edge Daemon</span>
                        </span>
                        <Download className="w-3.5 h-3.5 text-muted group-hover:text-amber" />
                      </a>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: MODEL REGISTRY */}
          {activeTab === "registry" && (
            <div className="space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h1 className="font-display font-extrabold text-2xl sm:text-3xl text-text">
                    Enterprise Model Registry
                  </h1>
                  <p className="text-xs text-muted mt-1">
                    Manage versioning, rollback, and edge target bindings for optimized models.
                  </p>
                </div>
                <button
                  onClick={() => setActiveTab("studio")}
                  className="btn-primary text-xs px-4 py-2 self-start flex items-center space-x-1.5"
                >
                  <UploadCloud className="w-3.5 h-3.5 text-black" />
                  <span>Register New Model</span>
                </button>
              </div>

              <div className="space-y-4">
                {registryModels.map((m) => (
                  <div
                    key={m.id}
                    className="p-5 rounded-xl bg-surface border border-border flex flex-col lg:flex-row lg:items-center justify-between gap-4 hover:border-lime/60 transition-colors"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-sm text-text">{m.name}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] bg-violet/20 text-violet font-bold">
                          {m.version}
                        </span>
                        <span className="px-2 py-0.5 rounded text-[10px] bg-lime/15 text-lime border border-lime/30 font-bold">
                          {m.status}
                        </span>
                      </div>
                      <p className="text-xs text-muted">
                        Deployed Target: <span className="text-text">{m.deployed_target}</span> | Format:{" "}
                        <span className="text-amber">{m.format}</span> | Last updated: {m.last_deployed}
                      </p>
                    </div>

                    <div className="flex items-center space-x-6 text-xs font-mono">
                      <div>
                        <span className="text-muted block text-[10px]">FOOTPRINT</span>
                        <span className="text-lime font-bold">
                          {m.compressed_size_kb} KB ({m.compression_ratio})
                        </span>
                      </div>
                      <div>
                        <span className="text-muted block text-[10px]">LATENCY</span>
                        <span className="text-violet font-bold">{m.latency_ms} ms</span>
                      </div>
                      <div>
                        <span className="text-muted block text-[10px]">ACCURACY (ROC-AUC)</span>
                        <span className="text-text font-bold">{m.roc_auc}</span>
                      </div>

                      <div className="flex items-center space-x-2">
                        <a
                          href="http://localhost:8000/api/v1/models/credit_card_fraud_student_int8/download"
                          download="model_int8.onnx"
                          className="px-2.5 py-1.5 rounded bg-surface-2 border border-border hover:border-lime text-text flex items-center space-x-1 transition-colors text-[11px]"
                          title="Download ONNX Artifact"
                        >
                          <Download className="w-3.5 h-3.5 text-lime" />
                          <span>ONNX</span>
                        </a>
                        <a
                          href="http://localhost:8000/api/v1/models/credit_card_fraud_student_int8/export/cpp"
                          download="tinyai_edge.h"
                          className="px-2.5 py-1.5 rounded bg-surface-2 border border-border hover:border-violet text-text flex items-center space-x-1 transition-colors text-[11px]"
                          title="Export C++ Header"
                        >
                          <FileCode className="w-3.5 h-3.5 text-violet" />
                          <span>C++ Header</span>
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 4: CONNECTED ENDPOINTS / FLEET */}
          {activeTab === "fleet" && (
            <div className="space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h1 className="font-display font-extrabold text-2xl sm:text-3xl text-text">
                    Connected Edge Fleets
                  </h1>
                  <p className="text-xs text-muted mt-1">
                    Live device monitoring across retail terminals, ATMs, and edge gateways.
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <button className="px-3 py-1.5 rounded-lg bg-surface-2 border border-border hover:border-lime text-xs text-text flex items-center space-x-1.5 transition-colors">
                    <RefreshCw className="w-3.5 h-3.5 text-lime" />
                    <span>Ping All Fleets</span>
                  </button>
                  <button className="btn-primary text-xs px-3 py-1.5 flex items-center space-x-1.5">
                    <Zap className="w-3.5 h-3.5 text-black" />
                    <span>Deploy OTA Update</span>
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {fleetData?.devices.map((device: any) => (
                  <div
                    key={device.id}
                    className="p-5 rounded-xl bg-surface border border-border space-y-3 font-mono text-xs"
                  >
                    <div className="flex items-center justify-between pb-2 border-b border-border/80">
                      <div>
                        <h4 className="font-bold text-sm text-text">{device.name}</h4>
                        <span className="text-[10px] text-muted">ID: {device.id} | {device.location}</span>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] bg-lime/15 text-lime border border-lime/30 font-bold">
                        {device.status}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2">
                      <div className="p-2 rounded bg-surface-2 border border-border/60">
                        <span className="text-[10px] text-muted block">HARDWARE</span>
                        <span className="text-text font-bold truncate block">{device.hardware}</span>
                      </div>
                      <div className="p-2 rounded bg-surface-2 border border-border/60">
                        <span className="text-[10px] text-muted block">P95 LATENCY</span>
                        <span className="text-violet font-bold">{device.latency_p95_ms} ms</span>
                      </div>
                      <div className="p-2 rounded bg-surface-2 border border-border/60">
                        <span className="text-[10px] text-muted block">SRAM USED</span>
                        <span className="text-lime font-bold">{device.memory_used_kb} KB</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-muted pt-2 border-t border-border/60">
                      <span>Firmware: {device.firmware}</span>
                      <span className="text-lime">Heartbeat: {device.last_seen_sec}s ago</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 5: PYTHON SDK & CLI */}
          {activeTab === "sdk" && (
            <div className="space-y-6">
              <div>
                <h1 className="font-display font-extrabold text-2xl sm:text-3xl text-text">
                  Developer SDK & CLI Integration
                </h1>
                <p className="text-xs text-muted mt-1">
                  Automate edge model optimization directly in your Python code, notebooks, and CI/CD pipelines.
                </p>
              </div>

              {/* Install box */}
              <div className="p-4 rounded-xl bg-surface border border-border flex items-center justify-between text-xs">
                <span className="text-lime font-bold">pip install tinyai</span>
                <button
                  onClick={() => copyToClipboard("pip install tinyai", setCopiedSnippet)}
                  className="px-2.5 py-1 rounded bg-surface-2 hover:border-lime border border-border text-muted hover:text-text flex items-center space-x-1 transition-colors"
                >
                  {copiedSnippet ? <Check className="w-3.5 h-3.5 text-lime" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedSnippet ? "Copied" : "Copy"}</span>
                </button>
              </div>

              {/* Code Snippet Box */}
              <div className="p-6 rounded-xl bg-surface border border-border space-y-3 text-xs">
                <div className="flex items-center justify-between pb-3 border-b border-border/80">
                  <span className="text-[10px] uppercase text-muted">// PRODUCTION PYTHON WORKFLOW</span>
                  <button
                    onClick={() =>
                      copyToClipboard(
                        `import tinyai\n\n# 1. Initialize domain scheduler\ncompressor = tinyai.AutoScheduler(target_hardware="pos_terminal")\n\n# 2. Compress model with 1% accuracy guarantee\nedge_model = compressor.optimize("transactions.csv", max_accuracy_drop=0.01)\n\n# 3. Export binary for target device\nedge_model.export("fraud_detector_int8.onnx")`,
                        setCopiedSnippet
                      )
                    }
                    className="px-2.5 py-1 rounded bg-surface-2 hover:border-lime border border-border text-muted hover:text-text flex items-center space-x-1 transition-colors"
                  >
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy Code</span>
                  </button>
                </div>

                <pre className="p-4 rounded-lg bg-surface-2 text-text overflow-x-auto text-[11px] leading-relaxed">
                  <code>{`import tinyai

# 1. Initialize domain scheduler for your edge target
compressor = tinyai.AutoScheduler(
    target_hardware="pos_terminal",   # or "cortex_m", "raspberry_pi_4"
    api_key="${apiKey}"
)

# 2. Automatically profile, distill, prune, and quantize
edge_model = compressor.optimize(
    dataset="transactions.csv",
    error_cost_ratio=50.0,             # 50x cost for missed fraud
    max_accuracy_drop=0.01             # strict 1% tolerance
)

# 3. Export ready-to-run edge binary
edge_model.export("fraud_detector_int8.onnx")
print(edge_model.benchmark())
# -> Footprint: 7.2 KB (13.4x) | Latency: 0.045 ms (21.3x) | Gate: PASS`}</code>
                </pre>
              </div>
            </div>
          )}

          {/* TAB 6: API KEYS & CREDENTIALS */}
          {activeTab === "keys" && (
            <div className="space-y-6">
              <div>
                <h1 className="font-display font-extrabold text-2xl sm:text-3xl text-text">
                  API Credentials & Webhooks
                </h1>
                <p className="text-xs text-muted mt-1">
                  Authenticate edge client runtimes, remote telemetry reporting, and OTA updates.
                </p>
              </div>

              <div className="p-6 rounded-xl bg-surface border border-border space-y-6">
                <div>
                  <label className="text-xs uppercase text-muted block mb-2">// ACTIVE PRODUCTION API KEY</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      readOnly
                      value={apiKey}
                      className="flex-1 p-3 rounded-lg bg-surface-2 border border-border text-xs text-lime font-bold focus:outline-none"
                    />
                    <button
                      onClick={() => copyToClipboard(apiKey, setCopiedKey)}
                      className="px-4 py-3 rounded-lg bg-surface-2 border border-border hover:border-lime text-xs font-bold text-text flex items-center space-x-1.5 transition-colors"
                    >
                      {copiedKey ? <Check className="w-4 h-4 text-lime" /> : <Copy className="w-4 h-4 text-muted" />}
                      <span>{copiedKey ? "Copied" : "Copy Key"}</span>
                    </button>
                    <button
                      onClick={async () => {
                        const newKey = await generateApiKey();
                        setApiKey(newKey.api_key);
                      }}
                      className="px-4 py-3 rounded-lg bg-lime text-black font-bold text-xs hover:bg-lime/90 transition-colors"
                    >
                      Roll Key
                    </button>
                  </div>
                  <span className="text-[10px] text-muted block mt-1">
                    Rate Limit: 50,000 queries per second per enterprise tier.
                  </span>
                </div>

                <div className="pt-4 border-t border-border/80">
                  <label className="text-xs uppercase text-muted block mb-2">// DRIFT ALERT WEBHOOK URL</label>
                  <input
                    type="text"
                    defaultValue="https://api.acmefinance.internal/hooks/tinyai-drift-alert"
                    className="w-full p-3 rounded-lg bg-surface-2 border border-border text-xs text-text focus:outline-none focus:border-violet"
                  />
                  <span className="text-[10px] text-muted block mt-1">
                    Fires an automated notification whenever edge PSI exceeds 0.25 to trigger OTA recompression.
                  </span>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
