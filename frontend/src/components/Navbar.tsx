"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Sparkles, Zap } from "lucide-react";

interface NavbarProps {
  onOpenCommand?: () => void;
  onOpenAssistant?: () => void;
}

export default function Navbar({ onOpenCommand, onOpenAssistant }: NavbarProps) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-[#0B0A10]/85 backdrop-blur-md border-b border-[#2A2638] py-3 shadow-lg"
          : "bg-transparent py-5"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        {/* Brand */}
        <a href="#hero" className="flex items-center space-x-3 group">
          <div className="w-10 h-10 rounded-lg bg-surface border border-border flex items-center justify-center relative overflow-hidden group-hover:border-lime transition-colors">
            <Zap className="w-5 h-5 text-lime animate-pulse" />
            <div className="absolute inset-0 bg-lime/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <span className="font-display font-extrabold text-2xl tracking-tight text-text">
            Tiny<span className="text-lime">AI</span>
          </span>
        </a>

        {/* Navigation Links */}
        <nav className="hidden lg:flex items-center space-x-6 text-sm font-medium">
          <a href="#overview" className="text-muted hover:text-text transition-colors">
            Overview
          </a>
          <a href="#pipeline" className="text-muted hover:text-text transition-colors">
            Pipeline
          </a>
          <a href="#scheduler" className="text-muted hover:text-text transition-colors flex items-center space-x-1">
            <span className="text-lime">★</span>
            <span>Scheduler</span>
          </a>
          <a href="#playground" className="text-muted hover:text-lime transition-colors">
            Playground
          </a>
          <a href="#benchmarks" className="text-muted hover:text-text transition-colors">
            Results
          </a>
          <a href="#edge-sim" className="text-muted hover:text-text transition-colors">
            Edge Simulator
          </a>
        </nav>

        {/* Right Actions */}
        <div className="flex items-center space-x-3">
          {/* AI Assistant button */}
          <button
            onClick={onOpenAssistant}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-surface border border-border hover:border-violet text-xs font-mono text-text transition-all hover:shadow-violet-glow"
            title="Ask AI Research Assistant"
          >
            <Sparkles className="w-3.5 h-3.5 text-violet" />
            <span className="hidden md:inline">AI Assistant</span>
          </button>

          {/* Command Palette Button */}
          <button
            onClick={onOpenCommand}
            className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-surface border border-border hover:border-lime text-xs font-mono text-muted hover:text-text transition-colors"
          >
            <kbd className="text-[10px] bg-surface-2 px-1 rounded border border-border">⌘K</kbd>
          </button>

          {/* Primary CTA */}
          <Link
            href="/console"
            className="btn-primary text-xs px-4 py-2 flex items-center space-x-1.5 shadow-lime-glow"
          >
            <Zap className="w-3.5 h-3.5 text-black" />
            <span>Launch Studio Console</span>
          </Link>
        </div>
      </div>
    </header>
  );
}
