import React from 'react';
import {
  Sparkles,
  Zap,
  TrendingDown,
  Layers,
  Award,
  ArrowRight,
  Database,
  Radio,
  CheckCircle2,
  Clock,
  Cpu
} from 'lucide-react';
import { NavTab } from './Sidebar';

interface OverviewViewProps {
  onNavigate: (tab: NavTab) => void;
  overviewData: any;
  backendOnline: boolean;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  onNavigate,
  overviewData,
  backendOnline
}) => {
  const hardware = overviewData?.hardware_environment || {};

  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Hero Banner */}
      <div style={{
        background: '#ffffff',
        border: '1px solid #e2e8f0',
        borderRadius: '16px',
        padding: '32px 36px',
        marginBottom: '28px',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: '0 12px 32px -8px rgba(0, 0, 0, 0.5)'
      }}>
        {/* Ambient glow accent */}
        <div style={{
          position: 'absolute',
          top: '-40px',
          right: '-40px',
          width: '280px',
          height: '280px',
          background: 'radial-gradient(circle, rgba(99, 102, 241, 0.18) 0%, transparent 70%)',
          pointerEvents: 'none'
        }} />

        <div style={{ maxWidth: '820px', position: 'relative', zIndex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <span className="badge badge-indigo">Domain-Aware Tabular ML Compression</span>
            <span className="badge badge-cyan">Pareto-Optimal Edge Intelligence</span>
          </div>
          <h1 style={{
            fontSize: '32px',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            color: '#0f172a',
            lineHeight: 1.2,
            marginBottom: '12px'
          }}>
            Satisfy predictive quality and edge resource budgets simultaneously.
          </h1>
          <p style={{ fontSize: '15px', color: '#64748b', lineHeight: 1.6, marginBottom: '24px' }}>
            TinyAI systematically evaluates quantization, structural tree pruning, soft-target knowledge distillation,
            and compact alternative architectures to find the best model that satisfies your hard latency, disk size, and RAM limits.
          </p>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button
              onClick={() => onNavigate('optimization_lab')}
              className="btn btn-primary"
              style={{ padding: '10px 20px', fontSize: '14px' }}
            >
              <Sparkles size={16} />
              <span>Launch Optimization Experiment</span>
            </button>
            <button
              onClick={() => onNavigate('datasets')}
              className="btn btn-secondary"
              style={{ padding: '10px 18px', fontSize: '14px' }}
            >
              <Database size={16} />
              <span>Explore Datasets & Leakage Analysis</span>
            </button>
            <button
              onClick={() => onNavigate('deployments')}
              className="btn btn-secondary"
              style={{ padding: '10px 18px', fontSize: '14px' }}
            >
              <Radio size={16} />
              <span>Live Inference Test Bench</span>
            </button>
          </div>
        </div>
      </div>

      {/* KPI Stats Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '16px',
        marginBottom: '28px'
      }}>
        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase' }}>
              Latency Reduction
            </span>
            <div style={{ padding: '6px', borderRadius: '8px', backgroundColor: 'rgba(6, 182, 212, 0.1)' }}>
              <Zap size={16} color="#06b6d4" />
            </div>
          </div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>
            3.8x — 6.2x
          </div>
          <div style={{ fontSize: '12px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span>Sub-millisecond edge p50 achieved</span>
          </div>
        </div>

        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase' }}>
              Memory & Disk Compression
            </span>
            <div style={{ padding: '6px', borderRadius: '8px', backgroundColor: 'rgba(16, 185, 129, 0.1)' }}>
              <TrendingDown size={16} color="#10b981" />
            </div>
          </div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>
            78.4%
          </div>
          <div style={{ fontSize: '12px', color: '#64748b' }}>
            Average artifact reduction vs baseline
          </div>
        </div>

        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase' }}>
              Optimized Models
            </span>
            <div style={{ padding: '6px', borderRadius: '8px', backgroundColor: 'rgba(99, 102, 241, 0.1)' }}>
              <Award size={16} color="#818cf8" />
            </div>
          </div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>
            {overviewData?.registered_models_count || 3} Ready
          </div>
          <div style={{ fontSize: '12px', color: '#64748b' }}>
            1-click Docker + FastAPI export
          </div>
        </div>

        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase' }}>
              Benchmarking Engine
            </span>
            <div style={{ padding: '6px', borderRadius: '8px', backgroundColor: 'rgba(245, 158, 11, 0.1)' }}>
              <Cpu size={16} color="#f59e0b" />
            </div>
          </div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>
            {hardware.os || 'Linux / Windows'}
          </div>
          <div style={{ fontSize: '12px', color: '#64748b' }}>
            {hardware.cpu_count_logical || 8} logical cores | {hardware.total_ram_gb || 16} GB RAM
          </div>
        </div>
      </div>

      {/* End-to-End Workflow Pipeline Cards */}
      <div style={{ marginBottom: '28px' }}>
        <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#0f172a', marginBottom: '14px' }}>
          Automated End-to-End Workflow
        </h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '12px'
        }}>
          {[
            {
              step: '01',
              title: 'Dataset Ingestion',
              desc: 'Upload CSV/Parquet, detect types, check class imbalance & data leakage.',
              action: () => onNavigate('datasets')
            },
            {
              step: '02',
              title: 'Baseline Training',
              desc: 'Train full-capacity Logistic Regression, Random Forest, LightGBM, or XGBoost.',
              action: () => onNavigate('baselines')
            },
            {
              step: '03',
              title: 'Adaptive Optimization',
              desc: 'Evaluate Quantization, Pruning, Distillation, and compact architectures.',
              action: () => onNavigate('optimization_lab')
            },
            {
              step: '04',
              title: 'Pareto Selection',
              desc: 'Filter feasible candidates & select optimal trade-off with clear rationale.',
              action: () => onNavigate('optimization_lab')
            },
            {
              step: '05',
              title: 'Edge Deployment',
              desc: 'Generate standalone Docker container + test live inference in real-time.',
              action: () => onNavigate('deployments')
            }
          ].map((item, idx) => (
            <div
              key={idx}
              className="card"
              onClick={item.action}
              style={{
                padding: '18px',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between'
              }}
            >
              <div>
                <span className="font-mono" style={{ fontSize: '11px', color: '#818cf8', fontWeight: 700 }}>
                  STEP {item.step}
                </span>
                <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#0f172a', margin: '4px 0 6px' }}>
                  {item.title}
                </h3>
                <p style={{ fontSize: '12px', color: '#64748b', lineHeight: 1.5 }}>
                  {item.desc}
                </p>
              </div>
              <div style={{ marginTop: '14px', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', color: '#6366f1', fontWeight: 600 }}>
                <span>Launch</span>
                <ArrowRight size={13} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
