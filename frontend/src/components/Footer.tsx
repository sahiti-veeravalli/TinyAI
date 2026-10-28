"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Copy, Check, Github, BookOpen, ShieldCheck, Activity, Terminal, ArrowUpRight } from "lucide-react";

export default function Footer() {
  const [copied, setCopied] = useState(false);

  const whitepaperCite = `@article{tinyai2026edge,
  title={TinyAI: Domain-Aware Model Compression and Deployment Pipeline for Enterprise Analytics Edge Devices},
  author={Veeravalli, Sahiti and Collaborators},
  year={2026},
  url={https://github.com/sahiti-veeravalli/TinyAI}
}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(whitepaperCite);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <footer id="platform-info" className="border-t border-border bg-[#0B0A10] py-16 text-xs font-mono text-muted">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        {/* Enterprise Platform Navigation Grid */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8">
          <div className="col-span-2 space-y-3">
            <div className="flex items-center space-x-2">
              <span className="font-display font-extrabold text-2xl text-text">
                Tiny<span className="text-lime">AI</span>
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] bg-lime/10 border border-lime/30 text-lime font-bold">
                ENTERPRISE
              </span>
            </div>
            <p className="text-muted leading-relaxed text-xs max-w-sm">
              The autonomous edge AI compression and deployment platform for tabular predictive systems. Compress deep neural networks up to 13.4x for ARM Cortex-M, ESP32, and POS edge gateways with zero cloud dependencies.
            </p>
            <div className="flex items-center space-x-3 pt-2">
              <a
                href="https://github.com/sahiti-veeravalli/TinyAI"
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-1.5 rounded-lg bg-surface border border-border hover:border-lime text-text flex items-center space-x-1.5 transition-colors"
              >
                <Github className="w-4 h-4" />
                <span>GitHub</span>
                <ArrowUpRight className="w-3 h-3 text-muted" />
              </a>
              <Link
                href="/console"
                className="px-3 py-1.5 rounded-lg bg-lime/10 border border-lime/30 text-lime hover:bg-lime/20 font-bold flex items-center space-x-1.5 transition-colors"
              >
                <Terminal className="w-3.5 h-3.5" />
                <span>Console</span>
              </Link>
            </div>
          </div>

          <div className="space-y-2.5">
            <span className="text-text font-bold uppercase block text-xs">// PLATFORM</span>
            <ul className="space-y-2 text-xs">
              <li>
                <Link href="/console" className="hover:text-text transition-colors">
                  Optimization Studio
                </Link>
              </li>
              <li>
                <Link href="/console" className="hover:text-text transition-colors">
                  Model Registry
                </Link>
              </li>
              <li>
                <Link href="/console" className="hover:text-text transition-colors">
                  Edge Fleet Telemetry
                </Link>
              </li>
              <li>
                <a href="#edge-sim" className="hover:text-text transition-colors">
                  Hardware Execution Profiler
                </a>
              </li>
              <li>
                <a href="#playground" className="hover:text-text transition-colors">
                  Live ONNX Inference Lab
                </a>
              </li>
            </ul>
          </div>

          <div className="space-y-2.5">
            <span className="text-text font-bold uppercase block text-xs">// EMBEDDED RUNTIMES</span>
            <ul className="space-y-2 text-xs">
              <li>
                <a
                  href="http://localhost:8000/api/v1/models/credit_card_fraud_student_int8/export/cpp"
                  className="hover:text-lime transition-colors flex items-center space-x-1"
                >
                  <span>C++11 Single-Header Runtime</span>
                </a>
              </li>
              <li>
                <a
                  href="http://localhost:8000/api/v1/models/credit_card_fraud_student_int8/export/python"
                  className="hover:text-lime transition-colors"
                >
                  Python Edge Daemon
                </a>
              </li>
              <li>
                <span className="text-muted">ARM CMSIS-NN Quantized</span>
              </li>
              <li>
                <span className="text-muted">TensorRT Edge Container</span>
              </li>
              <li>
                <span className="text-muted">ESP-IDF Microcontroller C</span>
              </li>
            </ul>
          </div>

          <div className="space-y-2.5">
            <span className="text-text font-bold uppercase block text-xs">// ENTERPRISE TRUST</span>
            <ul className="space-y-2 text-xs">
              <li className="flex items-center space-x-1 text-text">
                <ShieldCheck className="w-3.5 h-3.5 text-lime" />
                <span>SOC2 Type II Aligned</span>
              </li>
              <li>Zero Data Egress (Air-Gapped)</li>
              <li>Deterministic Microsecond SLA</li>
              <li>
                <button
                  onClick={handleCopy}
                  className="text-muted hover:text-violet flex items-center space-x-1 transition-colors"
                >
                  <BookOpen className="w-3 h-3" />
                  <span>{copied ? "Citation Copied" : "Architecture Reference"}</span>
                </button>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar with Status & Copyright */}
        <div className="border-t border-border/60 pt-6 flex flex-col sm:flex-row items-center justify-between text-[11px] gap-4">
          <div className="flex items-center space-x-3">
            <span className="flex items-center space-x-1.5 text-lime">
              <span className="w-2 h-2 rounded-full bg-lime animate-pulse" />
              <span>All Systems Operational (100% Edge Uptime)</span>
            </span>
            <span className="text-border">|</span>
            <span>Apache 2.0 Open Architecture</span>
          </div>
          <p>© 2026 TinyAI Inc. Enterprise Edge Predictive Platform.</p>
        </div>
      </div>
    </footer>
  );
}
