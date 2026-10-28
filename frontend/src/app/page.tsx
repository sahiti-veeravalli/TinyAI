"use client";

import React, { useState } from "react";
import Navbar from "@/components/Navbar";
import Hero3D from "@/components/Hero3D";
import ProblemSection from "@/components/ProblemSection";
import PipelineFlow from "@/components/PipelineFlow";
import DomainScheduler from "@/components/DomainScheduler";
import Playground from "@/components/Playground";
import ResultsBenchmarks from "@/components/ResultsBenchmarks";
import EdgeSimulator from "@/components/EdgeSimulator";
import ExplainabilityTrust from "@/components/ExplainabilityTrust";
import DriftMonitor from "@/components/DriftMonitor";
import Footer from "@/components/Footer";
import AIAssistant from "@/components/AIAssistant";
import CommandPalette from "@/components/CommandPalette";

export default function Home() {
  const [isCommandOpen, setIsCommandOpen] = useState(false);
  const [isAssistantOpen, setIsAssistantOpen] = useState(false);

  return (
    <main className="min-h-screen bg-bg text-text relative selection:bg-magenta selection:text-white">
      {/* Navigation */}
      <Navbar
        onOpenCommand={() => setIsCommandOpen(true)}
        onOpenAssistant={() => setIsAssistantOpen(true)}
      />

      {/* 1. Hero with 3D shrinking model and live counters */}
      <Hero3D />

      {/* 2. The Problem: Cloud vs Edge constraints */}
      <ProblemSection />

      {/* 3. The Interactive Pipeline flow diagram */}
      <PipelineFlow />

      {/* 4. The Domain-Aware Compression Scheduler (Patentable Core) */}
      <DomainScheduler />

      {/* 5. Playground Wizard & In-Browser Dual Inference */}
      <Playground />

      {/* 6. Results and Ablation Benchmarks */}
      <ResultsBenchmarks />

      {/* 7. Edge Device Hardware Simulator */}
      <EdgeSimulator />

      {/* 8. Explainability (SHAP) & Fairness Auditing */}
      <ExplainabilityTrust />

      {/* 9. Live Drift Monitoring */}
      <DriftMonitor />

      {/* 10. Footer with IEEE Access BibTeX citation */}
      <Footer />

      {/* Overlays */}
      <CommandPalette
        isOpen={isCommandOpen}
        onClose={() => setIsCommandOpen(false)}
        onOpenAssistant={() => setIsAssistantOpen(true)}
      />

      <AIAssistant
        isOpen={isAssistantOpen}
        onClose={() => setIsAssistantOpen(false)}
      />
    </main>
  );
}
