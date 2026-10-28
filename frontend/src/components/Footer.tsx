"use client";

import React, { useState } from "react";
import { Copy, Check, Github, BookOpen, Shield, Mail } from "lucide-react";

export default function Footer() {
  const [copied, setCopied] = useState(false);

  const bibtex = `@article{tinyai2026edge,
  title={TinyAI: Quantization, Pruning, and Distillation Pipeline for Enterprise Analytics Edge Devices},
  author={Veeravalli, Sahiti and Collaborators},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <footer id="paper" className="border-t border-border bg-[#0B0A10] py-16 text-xs font-mono text-muted">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        {/* BibTeX Citation Box */}
        <div className="glass-card p-6 border border-border">
          <div className="flex items-center justify-between pb-3 border-b border-border/70 mb-4">
            <div className="flex items-center space-x-2">
              <BookOpen className="w-4 h-4 text-violet" />
              <span className="font-bold text-text uppercase">// CITE THIS WORK (IEEE ACCESS FORMAT)_</span>
            </div>
            <button
              onClick={handleCopy}
              className="px-3 py-1 rounded bg-surface-2 border border-border hover:border-lime text-text flex items-center space-x-1 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-lime" />
                  <span className="text-lime">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-muted" />
                  <span>Copy BibTeX</span>
                </>
              )}
            </button>
          </div>

          <pre className="p-4 rounded-lg bg-surface-2 border border-border/60 text-muted overflow-x-auto text-[11px] leading-relaxed">
            <code>{bibtex}</code>
          </pre>
        </div>

        {/* Footer Links & Credits */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 pt-4">
          <div className="space-y-2">
            <span className="font-display font-extrabold text-lg text-text">
              Tiny<span className="text-lime">AI</span>
            </span>
            <p className="text-muted leading-relaxed">
              Brings big-model accuracy to edge predictive devices. Built for IEEE Access submission & patent examination.
            </p>
          </div>

          <div className="space-y-2">
            <span className="text-text font-bold uppercase block">// RESEARCH VENUE</span>
            <p>Target: IEEE Access (Regular Paper)</p>
            <p>Track: Industrial Cyber-Physical Systems & Edge AI</p>
            <p>Review Readiness: Camera-Ready Demonstration</p>
          </div>

          <div className="space-y-2">
            <span className="text-text font-bold uppercase block">// INTELLECTUAL PROPERTY</span>
            <p>Patent Claim: Domain-Aware Compression Scheduling for Edge Tabular AI</p>
            <p>Priority Date: October 2026</p>
            <p className="text-[10px]">Disclaimer: Research presentation; not legal advice.</p>
          </div>

          <div className="space-y-2">
            <span className="text-text font-bold uppercase block">// REPRODUCIBILITY</span>
            <p>Code License: Apache 2.0</p>
            <p>Datasets: Kaggle, UCI Machine Learning, OpenML</p>
            <a href="https://github.com" className="text-lime hover:underline flex items-center space-x-1">
              <Github className="w-3.5 h-3.5" />
              <span>GitHub Repository</span>
            </a>
          </div>
        </div>

        <div className="border-t border-border/60 pt-6 flex flex-col sm:flex-row items-center justify-between text-[11px]">
          <p>© 2026 TinyAI Research Team. All rights reserved.</p>
          <p className="text-muted">
            Designed with <span className="text-magenta font-bold">Voltage Punk</span> Aesthetics.
          </p>
        </div>
      </div>
    </footer>
  );
}
