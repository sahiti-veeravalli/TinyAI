"use client";

import React, { useRef, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, FileText, Zap, ShieldCheck, Flame, CheckCircle2 } from "lucide-react";

export default function Hero3D() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [modelSizeKb, setModelSizeKb] = useState(96.5);
  const [isCompressed, setIsCompressed] = useState(false);
  const [activeNodesCount, setActiveNodesCount] = useState(48);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 500);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 450);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener("resize", handleResize);

    // Generate neural network nodes
    interface Node {
      x: number;
      y: number;
      vx: number;
      vy: number;
      layer: number;
      isCore: boolean;
      alive: boolean;
      sparkY: number;
      sparkVy: number;
      flash: number;
      color: string;
    }

    const layers = [4, 7, 10, 8, 5, 2];
    const nodes: Node[] = [];
    const totalLayers = layers.length;

    layers.forEach((count, lIdx) => {
      const layerX = (width / (totalLayers + 1)) * (lIdx + 1);
      for (let i = 0; i < count; i++) {
        const spacing = height / (count + 1);
        const isCore = lIdx === 0 || lIdx === totalLayers - 1 || (i % 2 === 0 && lIdx >= 2 && lIdx <= 4);
        nodes.push({
          x: layerX + (Math.random() - 0.5) * 20,
          y: spacing * (i + 1) + (Math.random() - 0.5) * 20,
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          layer: lIdx,
          isCore,
          alive: true,
          sparkY: 0,
          sparkVy: 0,
          flash: 0,
          color: "#7B5CFF", // Violet default
        });
      }
    });

    // Sparks array
    interface Spark {
      x: number;
      y: number;
      vx: number;
      vy: number;
      alpha: number;
      color: string;
    }
    const sparks: Spark[] = [];

    // Trigger intermittent pruning events
    const pruneInterval = setInterval(() => {
      // Find non-core living nodes
      const prunable = nodes.filter((n) => n.alive && !n.isCore);
      if (prunable.length > 0) {
        const target = prunable[Math.floor(Math.random() * prunable.length)];
        target.flash = 1.0;
        target.color = "#FF2E93"; // Hot magenta flash

        // Spawn sparks
        for (let s = 0; s < 6; s++) {
          sparks.push({
            x: target.x,
            y: target.y,
            vx: (Math.random() - 0.5) * 3,
            vy: Math.random() * 2 + 1,
            alpha: 1.0,
            color: "#FF2E93",
          });
        }

        setTimeout(() => {
          target.alive = false;
          setActiveNodesCount((prev) => Math.max(16, prev - 1));
          setModelSizeKb((prev) => {
            const next = Math.max(7.2, prev - 2.8);
            if (next <= 10.0) setIsCompressed(true);
            return Math.round(next * 10) / 10;
          });
        }, 180);
      } else {
        // Reset cycle
        nodes.forEach((n) => {
          n.alive = true;
          n.color = n.isCore ? "#C6FF3D" : "#7B5CFF";
        });
        setActiveNodesCount(nodes.length);
        setModelSizeKb(96.5);
        setIsCompressed(false);
      }
    }, 450);

    let frame = 0;
    const render = () => {
      frame++;
      ctx.clearRect(0, 0, width, height);

      // Draw connection edges
      ctx.lineWidth = 1;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const n1 = nodes[i];
          const n2 = nodes[j];
          if (n1.alive && n2.alive && n2.layer === n1.layer + 1) {
            const dist = Math.hypot(n1.x - n2.x, n1.y - n2.y);
            if (dist < 150) {
              const alpha = (1 - dist / 150) * (n1.isCore && n2.isCore ? 0.6 : 0.2);
              ctx.strokeStyle = n1.isCore && n2.isCore ? `rgba(198, 255, 61, ${alpha})` : `rgba(123, 92, 255, ${alpha})`;
              ctx.beginPath();
              ctx.moveTo(n1.x, n1.y);
              ctx.lineTo(n2.x, n2.y);
              ctx.stroke();
            }
          }
        }
      }

      // Draw sparks
      for (let s = sparks.length - 1; s >= 0; s--) {
        const spark = sparks[s];
        spark.x += spark.vx;
        spark.y += spark.vy;
        spark.alpha -= 0.025;
        if (spark.alpha <= 0) {
          sparks.splice(s, 1);
          continue;
        }
        ctx.fillStyle = `rgba(255, 46, 147, ${spark.alpha})`;
        ctx.fillRect(spark.x, spark.y, 2.5, 2.5);
      }

      // Draw nodes
      nodes.forEach((node) => {
        if (!node.alive) return;
        node.x += node.vx;
        node.y += node.vy;

        const radius = node.isCore ? 4.5 + Math.sin(frame * 0.06) * 0.8 : 3.5;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);

        if (node.flash > 0) {
          ctx.fillStyle = "#FF2E93";
          ctx.shadowColor = "#FF2E93";
          ctx.shadowBlur = 15;
          node.flash -= 0.05;
        } else if (node.isCore) {
          ctx.fillStyle = "#C6FF3D";
          ctx.shadowColor = "#C6FF3D";
          ctx.shadowBlur = 10;
        } else {
          ctx.fillStyle = "#7B5CFF";
          ctx.shadowColor = "#7B5CFF";
          ctx.shadowBlur = 6;
        }

        ctx.fill();
        ctx.shadowBlur = 0;
      });

      animationId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationId);
      clearInterval(pruneInterval);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <section id="hero" className="relative pt-32 pb-20 md:pt-40 md:pb-32 overflow-hidden bg-dot-grid">
      {/* Scanline hero effect */}
      <div className="scanlines absolute inset-0 pointer-events-none" />

      {/* Background radial gradient glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[650px] h-[450px] bg-gradient-to-tr from-violet/20 via-magenta/15 to-lime/10 blur-[130px] rounded-full pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Column: Headline and Microcopy */}
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-surface-2 border border-border">
              <span className="w-2 h-2 rounded-full bg-lime animate-ping" />
              <span className="text-xs font-mono text-lime font-semibold uppercase tracking-wider">
                Domain-Aware Compression Pipeline
              </span>
              <span className="text-xs font-mono text-muted">// v1.0 Production_</span>
            </div>

            <h1 className="font-display text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.05] text-text">
              Big-model brains.{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet via-magenta to-lime">
                Tiny-model body.
              </span>
            </h1>

            <p className="text-lg sm:text-xl text-muted font-normal max-w-2xl leading-relaxed">
              Brings high-capacity AI to resource-constrained enterprise analytics edge devices (POS, ATMs, IoT) while mathematically safeguarding precision within{" "}
              <span className="text-lime font-mono font-bold">1–2%</span> of the uncompressed cloud model.
            </p>

            {/* Real Measured Metric Counters */}
            <div className="grid grid-cols-3 gap-4 pt-4 border-y border-border/80 py-4 max-w-xl">
              <div>
                <p className="text-[11px] font-mono uppercase text-muted">// FOOTPRINT REDUCTION</p>
                <p className="text-3xl font-mono font-bold text-lime">13.4x</p>
                <p className="text-xs text-muted">96.5 KB → 7.2 KB</p>
              </div>
              <div>
                <p className="text-[11px] font-mono uppercase text-muted">// EDGE SPEEDUP</p>
                <p className="text-3xl font-mono font-bold text-violet">21.3x</p>
                <p className="text-xs text-muted">24.5 ms → 1.15 ms</p>
              </div>
              <div>
                <p className="text-[11px] font-mono uppercase text-muted">// ACCURACY RETAINED</p>
                <p className="text-3xl font-mono font-bold text-text">99.2%</p>
                <div className="inline-flex items-center space-x-1 text-[11px] text-lime font-mono">
                  <CheckCircle2 className="w-3 h-3 text-lime" />
                  <span>PASS GATE</span>
                </div>
              </div>
            </div>

            {/* CTAs */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <Link
                href="/console"
                className="btn-primary px-6 py-3.5 text-sm sm:text-base flex items-center space-x-2"
              >
                <span>Launch Studio Console</span>
                <ArrowRight className="w-4 h-4 text-black" />
              </Link>
              <a
                href="#pipeline"
                className="btn-secondary px-6 py-3.5 text-sm sm:text-base flex items-center space-x-2"
              >
                <FileText className="w-4 h-4 text-violet" />
                <span>Explore Architecture</span>
              </a>
            </div>

            <p className="text-xs font-mono text-muted flex items-center space-x-2">
              <span className="text-lime">⚡</span>
              <span>In-browser inference powered by ONNX Runtime Web. Zero data leaves your browser.</span>
            </p>
          </div>

          {/* Right Column: Dynamic Shrinking Model 3D Canvas */}
          <div className="lg:col-span-5 relative">
            <div className="glass-card p-6 relative overflow-hidden">
              <div className="flex items-center justify-between pb-3 border-b border-border/70 mb-4">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-violet" />
                  <span className="text-xs font-mono uppercase text-muted">// LIVE_TOPOLOGY_VIEW</span>
                </div>
                <div className="flex items-center space-x-2 font-mono text-xs">
                  <span className="text-muted">Nodes:</span>
                  <span className="text-magenta font-bold">{activeNodesCount} active</span>
                </div>
              </div>

              {/* Canvas visualizer */}
              <div className="w-full h-[320px] sm:h-[360px] relative rounded-lg bg-surface-2/60 border border-border flex items-center justify-center">
                <canvas ref={canvasRef} className="w-full h-full" />

                {/* Floating size ticker card */}
                <div className="absolute bottom-4 left-4 right-4 bg-surface/90 backdrop-blur-md border border-border p-3 rounded-lg flex items-center justify-between font-mono text-xs">
                  <div>
                    <span className="text-muted block text-[10px]">CURRENT MODEL SIZE</span>
                    <span className="text-lg font-bold text-lime">{modelSizeKb.toFixed(1)} KB</span>
                  </div>
                  <div className="text-right">
                    <span className="text-muted block text-[10px]">STATUS</span>
                    <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${isCompressed ? "bg-lime/20 text-lime" : "bg-magenta/20 text-magenta animate-pulse"}`}>
                      {isCompressed ? "Compressed. Smug about it." : "Pruning the fluff..."}
                    </span>
                  </div>
                </div>
              </div>

              {/* Legend */}
              <div className="flex items-center justify-around pt-4 text-[11px] font-mono text-muted">
                <div className="flex items-center space-x-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-violet" />
                  <span>Teacher Model</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-magenta" />
                  <span>Pruned Weights</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-lime" />
                  <span>Edge Student</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
