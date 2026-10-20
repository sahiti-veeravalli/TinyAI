import React, { useState, useEffect } from 'react';
import { BookOpen, Cpu, ShieldCheck, Zap, GitBranch, Layers } from 'lucide-react';
import { fetchHardwarePresets } from '../api';
import { HardwarePreset } from '../types';

export const DocumentationView: React.FC = () => {
  const [presets, setPresets] = useState<HardwarePreset[]>([]);

  useEffect(() => {
    fetchHardwarePresets().then(setPresets);
  }, []);

  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a' }}>
              Platform Architecture & Hardware Presets
            </h1>
            <span className="badge badge-indigo">Specification</span>
          </div>
          <p style={{ fontSize: '13px', color: '#64748b' }}>
            Reference documentation for edge runtimes, compression adapters, and Pareto frontier mathematics.
          </p>
        </div>
      </div>

      {/* Edge Hardware Presets Grid */}
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '14px' }}>
          Calibrated Hardware Runtime Presets
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          {presets.map((p) => (
            <div key={p.preset_id} className="card" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <Cpu size={16} color="#06b6d4" />
                <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a' }}>{p.name}</h3>
              </div>
              <p style={{ fontSize: '12px', color: '#64748b', marginBottom: '12px', minHeight: '36px' }}>
                {p.target_cpu}
              </p>
              <div style={{
                backgroundColor: 'rgba(0, 0, 0, 0.3)',
                padding: '10px 12px',
                borderRadius: '6px',
                marginBottom: '12px',
                fontSize: '11px',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                  <span>Max Disk Budget:</span>
                  <strong className="font-mono" style={{ color: '#818cf8' }}>{p.recommended_max_size_kb} KB</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                  <span>Target Latency SLA:</span>
                  <strong className="font-mono" style={{ color: '#22d3ee' }}>&lt; {p.recommended_max_latency_ms} ms</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                  <span>RAM Envelope:</span>
                  <strong className="font-mono" style={{ color: '#34d399' }}>{p.recommended_max_memory_mb} MB</strong>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                {p.supported_runtimes.map((r, i) => (
                  <span key={i} className="badge badge-indigo" style={{ fontSize: '9px' }}>
                    {r}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Technical Methodology Breakdown */}
      <div className="card" style={{ padding: '24px' }}>
        <h2 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '16px' }}>
          Compression Adapters & Limitations Declaration
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 700, color: '#818cf8', marginBottom: '4px' }}>
              1. Quantization Adapter
            </h3>
            <p style={{ fontSize: '12px', color: '#64748b', lineHeight: 1.6 }}>
              Quantizes neural network weights and biases to 16-bit half-precision float or dynamic int8.
              For tree ensembles (LightGBM/XGBoost), applies float16 leaf precision reduction and split threshold binning.
            </p>
          </div>

          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 700, color: '#06b6d4', marginBottom: '4px' }}>
              2. Structural Pruning Adapter
            </h3>
            <p style={{ fontSize: '12px', color: '#64748b', lineHeight: 1.6 }}>
              Trims low-information sub-trees in Random Forest and constrains boosting rounds/depth in LightGBM and XGBoost.
              Applies L1 unstructured magnitude pruning to neural networks to induce true parameter sparsity.
            </p>
          </div>

          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 700, color: '#10b981', marginBottom: '4px' }}>
              3. Distillation Adapter
            </h3>
            <p style={{ fontSize: '12px', color: '#64748b', lineHeight: 1.6 }}>
              Transfers ensemble dark knowledge from high-capacity teachers into ultra-compact student networks
              using temperature-scaled soft targets (T=2.0-3.0) and combined cross-entropy loss.
            </p>
          </div>

          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 700, color: '#f59e0b', marginBottom: '4px' }}>
              4. Compact Model Selection
            </h3>
            <p style={{ fontSize: '12px', color: '#64748b', lineHeight: 1.6 }}>
              Under strict micro-budgets (&lt;50KB), explores regularized L1 sparse linear models and shallow decision trees,
              which often outperform complex compressed deep architectures.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
