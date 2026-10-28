"use client";

import React, { useState, useEffect } from "react";
import { Search, Zap, BookOpen, Shield, Download, X } from "lucide-react";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenAssistant: () => void;
}

export default function CommandPalette({ isOpen, onClose, onOpenAssistant }: CommandPaletteProps) {
  const [search, setSearch] = useState("");

  const actions = [
    { label: "Go to Live Playground", href: "#playground", icon: Zap, cat: "Navigation" },
    { label: "View Domain-Aware Scheduler (Patent)", href: "#scheduler", icon: Shield, cat: "Core" },
    { label: "Explore Empirical Benchmarks & Ablation", href: "#benchmarks", icon: BookOpen, cat: "Research" },
    { label: "Open Edge Hardware Simulator", href: "#edge-sim", icon: Zap, cat: "Tools" },
    { label: "Inspect SHAP & Fairness Preservation", href: "#trust", icon: Shield, cat: "Trust" },
    { label: "Download Pre-trained INT8 ONNX Model", href: "http://localhost:8000/api/v1/models/credit_card_fraud_student_int8/download", icon: Download, cat: "Download" },
  ];

  const filtered = actions.filter((a) =>
    a.label.toLowerCase().includes(search.toLowerCase())
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (isOpen) onClose();
      }
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-start justify-center pt-24 px-4 font-mono text-xs">
      <div className="w-full max-w-xl bg-surface border border-border rounded-xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <div className="p-3 border-b border-border flex items-center space-x-2 bg-surface-2/60">
          <Search className="w-4 h-4 text-muted" />
          <input
            type="text"
            autoFocus
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search commands, benchmarks, or documentation..."
            className="flex-1 bg-transparent text-text placeholder-muted focus:outline-none"
          />
          <kbd className="px-1.5 py-0.5 rounded bg-surface border border-border text-[10px] text-muted">
            ESC
          </kbd>
        </div>

        <div className="max-h-72 overflow-y-auto p-2 space-y-1">
          {filtered.map((action, idx) => {
            const Icon = action.icon;
            return (
              <a
                key={idx}
                href={action.href}
                onClick={onClose}
                className="flex items-center justify-between p-2.5 rounded-lg hover:bg-surface-2 hover:border-lime border border-transparent transition-colors group"
              >
                <div className="flex items-center space-x-2.5">
                  <Icon className="w-4 h-4 text-violet group-hover:text-lime transition-colors" />
                  <span className="text-text group-hover:text-lime">{action.label}</span>
                </div>
                <span className="text-[10px] text-muted">{action.cat}</span>
              </a>
            );
          })}

          <button
            onClick={() => {
              onClose();
              onOpenAssistant();
            }}
            className="w-full flex items-center justify-between p-2.5 rounded-lg hover:bg-surface-2 hover:border-violet border border-transparent transition-colors text-left"
          >
            <div className="flex items-center space-x-2.5">
              <Zap className="w-4 h-4 text-violet" />
              <span className="text-text">Ask AI Research Assistant (RAG)</span>
            </div>
            <span className="text-[10px] text-violet">AI Proxy</span>
          </button>
        </div>
      </div>
    </div>
  );
}
