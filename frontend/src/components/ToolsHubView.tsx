import React, { useState } from 'react';
import {
  Minimize2,
  Sliders,
  Scissors,
  GraduationCap,
  Activity,
  TrendingDown,
  Box,
  ShieldAlert,
  Compass,
  PlayCircle,
  Eye,
  Award,
  Search,
  Sparkles,
  ArrowRight,
  Zap,
  CheckCircle2,
  Cpu,
  DownloadCloud
} from 'lucide-react';
import { ToolDefinition } from '../types';

export const AI_TOOLS: ToolDefinition[] = [
  {
    id: 'compress',
    name: 'Compress AI Model',
    shortDesc: 'Automated multi-objective compression to fit strict edge latency & size budgets.',
    badge: 'Multi-Objective',
    category: 'compression',
    icon: 'Minimize2',
    color: '#e5322d',
    bgLight: '#fef2f2',
    popular: true
  },
  {
    id: 'quantize',
    name: 'Quantize Precision',
    shortDesc: 'Convert split thresholds and leaf weights to Float16 / dynamic Int8 precision.',
    badge: 'FP16 Precision',
    category: 'compression',
    icon: 'Sliders',
    color: '#059669',
    bgLight: '#ecfdf5',
    popular: true
  },
  {
    id: 'prune',
    name: 'Prune Ensemble',
    shortDesc: 'Trim redundant tree depth and enforce structural L1 sparsity for reduced memory and latency.',
    badge: 'Structural Sparsity',
    category: 'compression',
    icon: 'Scissors',
    color: '#7c3aed',
    bgLight: '#f5f3ff',
    popular: true
  },
  {
    id: 'distill',
    name: 'Distill Student',
    shortDesc: 'Train a compact, lightweight student neural network guided by ensemble teacher logits.',
    badge: 'Logit Distillation',
    category: 'compression',
    icon: 'GraduationCap',
    color: '#d97706',
    bgLight: '#fffbeb'
  },
  {
    id: 'benchmark',
    name: 'Benchmark Lab',
    shortDesc: 'Measure real microsecond p50/p95/p99 latency, RAM peak, and throughput on host hardware.',
    badge: 'Microseconds',
    category: 'profiling',
    icon: 'Activity',
    color: '#2563eb',
    bgLight: '#eff6ff',
    popular: true
  },
  {
    id: 'pareto',
    name: 'Pareto Explorer',
    shortDesc: 'Interactive scatter plot of quality vs latency vs size. Pick non-dominated candidates.',
    badge: 'Pareto Frontier',
    category: 'profiling',
    icon: 'TrendingDown',
    color: '#4f46e5',
    bgLight: '#eef2ff'
  },
  {
    id: 'docker',
    name: 'Model to Docker',
    shortDesc: 'Export standalone FastAPI microservice, Dockerfile, preprocessor, and verification manifest.',
    badge: 'Production Bundle',
    category: 'deployment',
    icon: 'Box',
    color: '#0891b2',
    bgLight: '#ecfeff',
    popular: true
  },
  {
    id: 'sandbox',
    name: 'Live Inference Sandbox',
    shortDesc: 'Send test payloads and measure real response latency and probability distributions.',
    badge: 'Real-Time Timing',
    category: 'deployment',
    icon: 'PlayCircle',
    color: '#ea580c',
    bgLight: '#fff7ed',
    popular: true
  },
  {
    id: 'audit',
    name: 'Audit Dataset & Leakage',
    shortDesc: 'Profile class imbalance, schema types, missingness, and detect target leakage vectors.',
    badge: 'Pre-flight Check',
    category: 'audit',
    icon: 'ShieldAlert',
    color: '#e11d48',
    bgLight: '#fff1f2'
  },
  {
    id: 'explainability',
    name: 'Explainability & Attribution',
    shortDesc: 'Audit feature attribution fidelity before & after compression to verify decision stability.',
    badge: 'Cosine Fidelity',
    category: 'audit',
    icon: 'Eye',
    color: '#ca8a04',
    bgLight: '#fefce8'
  },
  {
    id: 'drift',
    name: 'Drift & Health Monitor',
    shortDesc: 'Track live inference drift with Kolmogorov-Smirnov test and Population Stability Index.',
    badge: 'KS & PSI Tests',
    category: 'profiling',
    icon: 'Compass',
    color: '#0d9488',
    bgLight: '#f0fdfa'
  },
  {
    id: 'registry',
    name: 'Model Registry',
    shortDesc: 'Manage versioned model artifacts, inspect performance cards, and track deployment status.',
    badge: 'Versioned Cards',
    category: 'deployment',
    icon: 'Award',
    color: '#475569',
    bgLight: '#f8fafc'
  }
];

interface ToolsHubViewProps {
  onSelectTool: (toolId: string) => void;
  onNavigateTab: (tab: any) => void;
}

export const ToolsHubView: React.FC<ToolsHubViewProps> = ({ onSelectTool, onNavigateTab }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const filteredTools = AI_TOOLS.filter((t) => {
    const matchesSearch =
      t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.shortDesc.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCat = selectedCategory === 'all' || t.category === selectedCategory;
    return matchesSearch && matchesCat;
  });

  const getIcon = (iconName: string, color: string) => {
    const props = { size: 28, color };
    switch (iconName) {
      case 'Minimize2': return <Minimize2 {...props} />;
      case 'Sliders': return <Sliders {...props} />;
      case 'Scissors': return <Scissors {...props} />;
      case 'GraduationCap': return <GraduationCap {...props} />;
      case 'Activity': return <Activity {...props} />;
      case 'TrendingDown': return <TrendingDown {...props} />;
      case 'Box': return <Box {...props} />;
      case 'PlayCircle': return <PlayCircle {...props} />;
      case 'ShieldAlert': return <ShieldAlert {...props} />;
      case 'Eye': return <Eye {...props} />;
      case 'Compass': return <Compass {...props} />;
      case 'Award': return <Award {...props} />;
      default: return <Sparkles {...props} />;
    }
  };

  return (
    <div className="animate-fade-in" style={{ backgroundColor: '#f8fafc', minHeight: 'calc(100vh - 64px)' }}>
      {/* Hero Section — iLovePDF style */}
      <section style={{
        padding: '54px 24px 36px',
        maxWidth: '1000px',
        margin: '0 auto',
        textAlign: 'center'
      }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 14px',
          borderRadius: '20px',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          marginBottom: '18px'
        }}>
          <Zap size={14} color="#e5322d" />
          <span style={{ fontSize: '13px', fontWeight: 700, color: '#e5322d' }}>
            Domain-Aware AI Model Compression for Edge & Micro-Runtimes
          </span>
        </div>

        <h1 style={{
          fontSize: '40px',
          fontWeight: 900,
          color: '#0f172a',
          letterSpacing: '-0.03em',
          lineHeight: 1.15,
          marginBottom: '16px'
        }}>
          Every tool you need to optimize AI models, in one place
        </h1>

        <p style={{
          fontSize: '18px',
          color: '#475569',
          maxWidth: '740px',
          margin: '0 auto 32px',
          lineHeight: 1.6
        }}>
          100% automated, budget-constrained compression. Reduce model size, cut microsecond latency,
          and package self-contained edge microservices with zero cloud lock-in.
        </p>

        {/* Search & Filter Bar */}
        <div style={{
          maxWidth: '640px',
          margin: '0 auto 24px',
          position: 'relative',
          display: 'flex',
          alignItems: 'center'
        }}>
          <Search size={18} color="#94a3b8" style={{ position: 'absolute', left: '16px' }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search AI tools (e.g. compress, quantize, prune, docker, benchmark)..."
            style={{
              width: '100%',
              padding: '14px 18px 14px 46px',
              fontSize: '15px',
              borderRadius: '30px',
              border: '1px solid #cbd5e1',
              backgroundColor: '#ffffff',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
              color: '#0f172a'
            }}
          />
        </div>

        {/* Category Pills */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {[
            { id: 'all', label: 'All AI Tools (12)' },
            { id: 'compression', label: 'Compression & Pruning (4)' },
            { id: 'profiling', label: 'Benchmarking & Profiling (3)' },
            { id: 'deployment', label: 'Deployment & Sandbox (3)' },
            { id: 'audit', label: 'Audit & Quality (2)' }
          ].map((cat) => (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.id)}
              style={{
                padding: '7px 16px',
                borderRadius: '20px',
                fontSize: '13px',
                fontWeight: 600,
                cursor: 'pointer',
                border: selectedCategory === cat.id ? '1px solid #e5322d' : '1px solid #e2e8f0',
                backgroundColor: selectedCategory === cat.id ? '#e5322d' : '#ffffff',
                color: selectedCategory === cat.id ? '#ffffff' : '#475569',
                transition: 'all 0.15s ease',
                boxShadow: selectedCategory === cat.id ? '0 2px 6px rgba(229, 50, 45, 0.25)' : 'none'
              }}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </section>

      {/* The Iconic iLove-Style Tool Cards Grid */}
      <section style={{
        maxWidth: '1320px',
        margin: '0 auto',
        padding: '0 24px 60px'
      }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '20px'
        }}>
          {filteredTools.map((tool) => {
            return (
              <div
                key={tool.id}
                onClick={() => onSelectTool(tool.id)}
                className="card"
                style={{
                  padding: '24px 22px',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  borderRadius: '14px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  position: 'relative',
                  overflow: 'hidden'
                }}
              >
                <div>
                  {/* Top Row: Icon Badge + Pill */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '18px' }}>
                    <div style={{
                      width: '54px',
                      height: '54px',
                      borderRadius: '12px',
                      backgroundColor: tool.bgLight,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: `1px solid ${tool.color}22`
                    }}>
                      {getIcon(tool.icon, tool.color)}
                    </div>
                    {tool.badge && (
                      <span style={{
                        fontSize: '11px',
                        fontWeight: 700,
                        color: tool.color,
                        backgroundColor: tool.bgLight,
                        padding: '3px 9px',
                        borderRadius: '12px',
                        border: `1px solid ${tool.color}33`,
                        letterSpacing: '0.01em'
                      }}>
                        {tool.badge}
                      </span>
                    )}
                  </div>

                  {/* Title & Description */}
                  <h3 style={{
                    fontSize: '18px',
                    fontWeight: 800,
                    color: '#0f172a',
                    marginBottom: '8px',
                    letterSpacing: '-0.02em'
                  }}>
                    {tool.name}
                  </h3>
                  <p style={{
                    fontSize: '13.5px',
                    color: '#64748b',
                    lineHeight: 1.5,
                    marginBottom: '20px'
                  }}>
                    {tool.shortDesc}
                  </p>
                </div>

                {/* Footer action */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  paddingTop: '12px',
                  borderTop: '1px solid #f1f5f9'
                }}>
                  <span style={{ fontSize: '13px', fontWeight: 700, color: tool.color, display: 'flex', alignItems: 'center', gap: '4px' }}>
                    Use Tool <ArrowRight size={14} />
                  </span>
                  <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'capitalize' }}>
                    {tool.category}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Trust & Proof Banner — iLovePDF style bottom highlight */}
      <section style={{
        backgroundColor: '#ffffff',
        borderTop: '1px solid #e2e8f0',
        padding: '48px 24px',
        textAlign: 'center'
      }}>
        <div style={{ maxWidth: '1080px', margin: '0 auto' }}>
          <h2 style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>
            The AI Compression Platform Engineered for Edge Physical Constraints
          </h2>
          <p style={{ fontSize: '15px', color: '#64748b', marginBottom: '36px' }}>
            Point-of-sale terminals, industrial sensors, and autonomous microcontrollers run on tight resource limits.
          </p>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '24px',
            textAlign: 'left'
          }}>
            <div style={{ padding: '20px', borderRadius: '12px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '8px', backgroundColor: '#fef2f2', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '12px' }}>
                <Minimize2 size={20} color="#e5322d" />
              </div>
              <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>2.5x to 30x Size Reduction</h4>
              <p style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.5 }}>
                Compress 200KB tabular ensembles down to 6KB - 70KB without losing critical predictive quality.
              </p>
            </div>

            <div style={{ padding: '20px', borderRadius: '12px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '8px', backgroundColor: '#ecfdf5', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '12px' }}>
                <Activity size={20} color="#059669" />
              </div>
              <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>Sub-0.30 ms Edge Inferences</h4>
              <p style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.5 }}>
                Pruned and quantized decision paths evaluate in microseconds on standard host CPU architectures.
              </p>
            </div>

            <div style={{ padding: '20px', borderRadius: '12px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '8px', backgroundColor: '#eff6ff', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '12px' }}>
                <DownloadCloud size={20} color="#2563eb" />
              </div>
              <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>1-Click Production Bundles</h4>
              <p style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.5 }}>
                Download a self-contained ZIP archive with standalone FastAPI service, Dockerfile, preprocessor, and model.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
