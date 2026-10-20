import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Sliders,
  Play,
  CheckCircle2,
  AlertCircle,
  TrendingDown,
  Zap,
  HardDrive,
  Award,
  Filter,
  Check,
  X,
  ExternalLink,
  Info
} from 'lucide-react';
import {
  ExperimentManifest,
  CandidateEvaluation,
  DatasetMetadata
} from '../types';
import {
  fetchDatasets,
  launchOptimization,
  fetchExperiments,
  exportModelBundle
} from '../api';

interface OptimizationLabViewProps {
  initialDatasetId?: string;
  initialBaselineFamily?: string;
  initialTargetCol?: string;
  onPromoteToRegistry?: (candidate: CandidateEvaluation) => void;
}

export const OptimizationLabView: React.FC<OptimizationLabViewProps> = ({
  initialDatasetId,
  initialBaselineFamily = 'lightgbm',
  initialTargetCol = 'is_fraud',
  onPromoteToRegistry
}) => {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(initialDatasetId || '');
  const [targetCol, setTargetCol] = useState<string>(initialTargetCol);
  const [baselineFamily, setBaselineFamily] = useState<string>(initialBaselineFamily);

  // Constraint state
  const [maxDiskSizeKb, setMaxDiskSizeKb] = useState<number>(250.0);
  const [maxLatencyMs, setMaxLatencyMs] = useState<number>(1.2);
  const [minQualityScore, setMinQualityScore] = useState<number>(0.85);
  const [priority, setPriority] = useState<string>('balanced');
  const [enabledStrategies, setEnabledStrategies] = useState<string[]>([
    'quantization',
    'pruning',
    'distillation',
    'model_selection'
  ]);

  // Execution state
  const [isOptimizing, setIsOptimizing] = useState<boolean>(false);
  const [activeStage, setActiveStage] = useState<string>('');
  const [manifest, setManifest] = useState<ExperimentManifest | null>(null);
  const [filterFeasibleOnly, setFilterFeasibleOnly] = useState<boolean>(false);
  const [hoveredCandidate, setHoveredCandidate] = useState<CandidateEvaluation | null>(null);
  const [exportedStatus, setExportedStatus] = useState<string | null>(null);

  useEffect(() => {
    fetchDatasets().then((d) => {
      setDatasets(d);
      if (d.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(d[0].dataset_id);
        const t = d[0].all_columns.includes('is_fraud') ? 'is_fraud' : d[0].all_columns[d[0].all_columns.length - 1];
        setTargetCol(t);
      }
    });

    fetchExperiments().then((exps) => {
      if (exps.length > 0) {
        setManifest(exps[0]);
      }
    });
  }, []);

  const handleLaunch = async () => {
    if (!selectedDatasetId) return;
    try {
      setIsOptimizing(true);
      setActiveStage('Benchmarking baseline model...');
      const res = await launchOptimization({
        dataset_id: selectedDatasetId,
        target_column: targetCol,
        baseline_family: baselineFamily,
        max_disk_size_kb: maxDiskSizeKb,
        max_latency_ms: maxLatencyMs,
        min_quality_score: minQualityScore,
        priority: priority,
        enabled_strategies: enabledStrategies
      });
      setManifest(res);
      setExportedStatus(null);
    } catch (e) {
      console.error(e);
    } finally {
      setIsOptimizing(false);
      setActiveStage('');
    }
  };

  const toggleStrategy = (strat: string) => {
    setEnabledStrategies((prev) =>
      prev.includes(strat) ? prev.filter((s) => s !== strat) : [...prev, strat]
    );
  };

  const candidates = manifest ? (filterFeasibleOnly ? manifest.candidates.filter((c) => c.is_feasible) : manifest.candidates) : [];
  const topCandidate = manifest?.recommended_candidate;
  const baselineCandidate = manifest?.candidates.find((c) => c.strategy === 'uncompressed_baseline');

  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1440px', margin: '0 auto' }}>
      {/* Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: 800, color: '#fff' }}>
              Optimization Lab & Pareto Frontier Engine
            </h1>
            <span className="badge badge-indigo">Multi-Objective Research</span>
          </div>
          <p style={{ fontSize: '13px', color: '#94a3b8' }}>
            Domain-aware, budget-constrained search across quantization, pruning, soft-target distillation, and model selection.
          </p>
        </div>
      </div>

      {/* Grid: Left Config Panel, Right Results / Pareto Plot */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '24px', marginBottom: '28px' }}>
        {/* Constraints & Strategy Configuration */}
        <div className="card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Sliders size={18} color="#818cf8" />
            <h2 style={{ fontSize: '15px', fontWeight: 700, color: '#fff' }}>
              Operational Constraints
            </h2>
          </div>

          {/* Dataset & Baseline Pickers */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
              Input Dataset
            </label>
            <select
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              style={{
                width: '100%',
                padding: '8px 10px',
                backgroundColor: '#0a0f1d',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '12px'
              }}
            >
              {datasets.map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.name} ({d.row_count.toLocaleString()} rows)
                </option>
              ))}
            </select>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
              Baseline Algorithm
            </label>
            <select
              value={baselineFamily}
              onChange={(e) => setBaselineFamily(e.target.value)}
              style={{
                width: '100%',
                padding: '8px 10px',
                backgroundColor: '#0a0f1d',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '12px'
              }}
            >
              <option value="lightgbm">LightGBM (Gradient Boosting)</option>
              <option value="xgboost">XGBoost (Extreme Gradient Boost)</option>
              <option value="random_forest">Random Forest (Bagged Trees)</option>
              <option value="tabular_mlp">Tabular MLP (Neural Network)</option>
              <option value="logistic_regression">Logistic Regression (Linear)</option>
            </select>
          </div>

          {/* Sliders for Hard Constraints */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#cbd5e1', marginBottom: '4px' }}>
              <span>Max Artifact Size (KB)</span>
              <strong className="font-mono" style={{ color: '#818cf8' }}>{maxDiskSizeKb} KB</strong>
            </div>
            <input
              type="range"
              min="20"
              max="1000"
              step="10"
              value={maxDiskSizeKb}
              onChange={(e) => setMaxDiskSizeKb(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: '#6366f1' }}
            />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#cbd5e1', marginBottom: '4px' }}>
              <span>Max p50 Latency (ms)</span>
              <strong className="font-mono" style={{ color: '#06b6d4' }}>{maxLatencyMs} ms</strong>
            </div>
            <input
              type="range"
              min="0.1"
              max="5.0"
              step="0.1"
              value={maxLatencyMs}
              onChange={(e) => setMaxLatencyMs(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: '#06b6d4' }}
            />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#cbd5e1', marginBottom: '4px' }}>
              <span>Min ROC-AUC / PR-AUC</span>
              <strong className="font-mono" style={{ color: '#10b981' }}>{minQualityScore}</strong>
            </div>
            <input
              type="range"
              min="0.70"
              max="0.99"
              step="0.01"
              value={minQualityScore}
              onChange={(e) => setMinQualityScore(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: '#10b981' }}
            />
          </div>

          {/* Priority selector */}
          <div style={{ marginBottom: '18px' }}>
            <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
              Optimization Priority Strategy
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
              {[
                { id: 'balanced', label: 'Balanced' },
                { id: 'lowest_latency', label: 'Lowest Latency' },
                { id: 'smallest_size', label: 'Smallest Size' },
                { id: 'highest_quality', label: 'Highest Quality' }
              ].map((p) => (
                <button
                  key={p.id}
                  onClick={() => setPriority(p.id)}
                  style={{
                    padding: '6px 8px',
                    borderRadius: '6px',
                    fontSize: '11px',
                    fontWeight: priority === p.id ? 600 : 500,
                    color: priority === p.id ? '#fff' : '#94a3b8',
                    backgroundColor: priority === p.id ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255, 255, 255, 0.03)',
                    border: priority === p.id ? '1px solid rgba(99, 102, 241, 0.5)' : '1px solid rgba(255, 255, 255, 0.06)',
                    cursor: 'pointer'
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Compression Adapters Toggles */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
              Active Compression Adapters
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {[
                { id: 'quantization', name: 'Quantization (Float16/Int8)' },
                { id: 'pruning', name: 'Ensemble Structural Pruning' },
                { id: 'distillation', name: 'Soft-Target Distillation' },
                { id: 'model_selection', name: 'Compact Model Exploration' }
              ].map((s) => {
                const active = enabledStrategies.includes(s.id);
                return (
                  <div
                    key={s.id}
                    onClick={() => toggleStrategy(s.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '7px 10px',
                      borderRadius: '6px',
                      backgroundColor: active ? 'rgba(99, 102, 241, 0.1)' : 'rgba(255, 255, 255, 0.02)',
                      border: active ? '1px solid rgba(99, 102, 241, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)',
                      cursor: 'pointer'
                    }}
                  >
                    <span style={{ fontSize: '11px', color: active ? '#fff' : '#64748b', fontWeight: active ? 600 : 400 }}>
                      {s.name}
                    </span>
                    <span style={{ fontSize: '12px', color: active ? '#818cf8' : '#475569' }}>
                      {active ? '✓' : '—'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <button
            onClick={handleLaunch}
            disabled={isOptimizing}
            className="btn btn-primary"
            style={{ width: '100%', padding: '10px' }}
          >
            <Play size={14} />
            <span>{isOptimizing ? 'Running Optimization Engine...' : 'Launch Optimization'}</span>
          </button>
        </div>

        {/* Right: Results, Pareto Frontier Chart & Recommended Model Banner */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Top Recommendation Highlight Card */}
          {topCandidate && (
            <div style={{
              background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%)',
              border: '1px solid rgba(16, 185, 129, 0.4)',
              borderRadius: '12px',
              padding: '20px 24px',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                    <span className="badge badge-green">Recommended Pareto Model</span>
                    <span className="badge badge-indigo">{topCandidate.strategy}</span>
                  </div>
                  <h3 style={{ fontSize: '18px', fontWeight: 800, color: '#fff' }}>
                    {topCandidate.name}
                  </h3>
                </div>

                <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase' }}>Quality ({topCandidate.quality_metric_name})</span>
                    <div className="font-mono" style={{ fontSize: '20px', fontWeight: 800, color: '#38bdf8' }}>
                      {topCandidate.quality_score.toFixed(4)}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase' }}>p50 Latency</span>
                    <div className="font-mono" style={{ fontSize: '20px', fontWeight: 800, color: '#22d3ee' }}>
                      {topCandidate.latency_p50_ms.toFixed(3)} ms
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase' }}>Disk Size</span>
                    <div className="font-mono" style={{ fontSize: '20px', fontWeight: 800, color: '#34d399' }}>
                      {topCandidate.disk_size_kb.toFixed(1)} KB
                    </div>
                  </div>
                </div>
              </div>

              {/* Rationale explanation */}
              <div style={{
                backgroundColor: 'rgba(0, 0, 0, 0.3)',
                padding: '12px 14px',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.05)',
                marginBottom: '14px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px', color: '#10b981', fontSize: '11px', fontWeight: 700 }}>
                  <Info size={13} />
                  <span>Transparent Selection Rationale</span>
                </div>
                <p style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.5 }}>
                  {topCandidate.recommendation_rationale}
                </p>
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                {onPromoteToRegistry && (
                  <button
                    onClick={() => onPromoteToRegistry(topCandidate)}
                    className="btn btn-primary btn-sm"
                  >
                    <Award size={14} />
                    <span>View in Model Registry</span>
                  </button>
                )}
                {baselineCandidate && (
                  <div style={{ fontSize: '12px', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>Vs Baseline:</span>
                    <strong style={{ color: '#34d399' }}>
                      {((1.0 - topCandidate.disk_size_kb / baselineCandidate.disk_size_kb) * 100).toFixed(0)}% smaller
                    </strong>
                    <span>•</span>
                    <strong style={{ color: '#22d3ee' }}>
                      {(baselineCandidate.latency_p50_ms / topCandidate.latency_p50_ms).toFixed(1)}x faster
                    </strong>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Interactive Pareto Frontier Chart (SVG Canvas) */}
          <div className="card" style={{ padding: '22px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#fff', marginBottom: '2px' }}>
                  Interactive Pareto Frontier (Quality vs Latency vs Size)
                </h3>
                <p style={{ fontSize: '11px', color: '#64748b' }}>
                  Hover over candidate markers to inspect trade-offs. Green halo denotes non-dominated Pareto optimal candidates.
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <label style={{ fontSize: '11px', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={filterFeasibleOnly}
                    onChange={(e) => setFilterFeasibleOnly(e.target.checked)}
                  />
                  <span>Feasible Only</span>
                </label>
              </div>
            </div>

            {/* Custom SVG Pareto Scatter Chart */}
            <div style={{
              width: '100%',
              height: '240px',
              backgroundColor: '#0a0f1d',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              position: 'relative',
              overflow: 'hidden'
            }}>
              {/* Axes Labels */}
              <div style={{ position: 'absolute', bottom: '8px', right: '14px', fontSize: '10px', color: '#64748b', fontWeight: 600 }}>
                Inference Latency p50 (ms) → Lower is better
              </div>
              <div style={{ position: 'absolute', top: '10px', left: '14px', fontSize: '10px', color: '#64748b', fontWeight: 600 }}>
                ↑ Predictive Quality (PR-AUC / ROC-AUC) Higher is better
              </div>

              {/* Threshold Boundary lines */}
              <div style={{
                position: 'absolute',
                top: `${(1.0 - (minQualityScore - 0.7) / 0.3) * 100}%`,
                left: 0,
                right: 0,
                borderTop: '1px dashed rgba(239, 68, 68, 0.35)',
                pointerEvents: 'none'
              }} />

              {/* Render Scatter Points */}
              <svg width="100%" height="100%" style={{ overflow: 'visible' }}>
                {candidates.map((cand, idx) => {
                  // Map Latency (0.1 to 2.5ms) to X (50px to 90%)
                  const minLat = 0.1;
                  const maxLat = 2.0;
                  const xNorm = Math.min(1.0, Math.max(0.0, (cand.latency_p50_ms - minLat) / (maxLat - minLat)));
                  const cx = 60 + xNorm * 460;

                  // Map Quality (0.7 to 1.0) to Y (top 20px to bottom 200px)
                  const minQ = 0.70;
                  const maxQ = 1.0;
                  const yNorm = Math.min(1.0, Math.max(0.0, (cand.quality_score - minQ) / (maxQ - minQ)));
                  const cy = 210 - yNorm * 180;

                  const isHovered = hoveredCandidate?.candidate_id === cand.candidate_id;
                  const isTop = topCandidate?.candidate_id === cand.candidate_id;

                  return (
                    <g
                      key={cand.candidate_id}
                      style={{ cursor: 'pointer' }}
                      onMouseEnter={() => setHoveredCandidate(cand)}
                      onMouseLeave={() => setHoveredCandidate(null)}
                    >
                      {/* Pareto Halo */}
                      {cand.is_pareto_optimal && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r={isTop ? 14 : 10}
                          fill="none"
                          stroke={isTop ? '#10b981' : '#6366f1'}
                          strokeWidth="2"
                          strokeDasharray={isTop ? 'none' : '3,3'}
                          opacity={0.8}
                        />
                      )}
                      {/* Main Node */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isTop ? 7 : 5}
                        fill={!cand.is_feasible ? '#ef4444' : isTop ? '#10b981' : '#818cf8'}
                        stroke="#fff"
                        strokeWidth={isHovered ? 2 : 1}
                      />
                      <text
                        x={cx + 9}
                        y={cy + 4}
                        fill={cand.is_feasible ? '#cbd5e1' : '#f87171'}
                        fontSize="10"
                        fontWeight={isTop ? '700' : '400'}
                      >
                        {cand.name}
                      </text>
                    </g>
                  );
                })}
              </svg>

              {/* Hover Tooltip Box */}
              {hoveredCandidate && (
                <div style={{
                  position: 'absolute',
                  top: '12px',
                  right: '12px',
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  borderRadius: '8px',
                  padding: '10px 14px',
                  fontSize: '11px',
                  boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
                  pointerEvents: 'none',
                  zIndex: 20
                }}>
                  <div style={{ fontWeight: 700, color: '#fff', marginBottom: '2px' }}>
                    {hoveredCandidate.name}
                  </div>
                  <div style={{ color: '#94a3b8' }}>Strategy: {hoveredCandidate.strategy}</div>
                  <div style={{ color: '#38bdf8' }}>Quality: {hoveredCandidate.quality_score.toFixed(4)}</div>
                  <div style={{ color: '#22d3ee' }}>Latency: {hoveredCandidate.latency_p50_ms.toFixed(3)} ms</div>
                  <div style={{ color: '#34d399' }}>Size: {hoveredCandidate.disk_size_kb.toFixed(1)} KB ({hoveredCandidate.compression_ratio.toFixed(1)}x)</div>
                  <div style={{ color: hoveredCandidate.is_feasible ? '#10b981' : '#ef4444', fontWeight: 600, marginTop: '4px' }}>
                    {hoveredCandidate.is_feasible ? '✓ Feasible' : `✗ Violations: ${hoveredCandidate.constraint_violations.join(', ')}`}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Candidate Comparison Table */}
      <div className="card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#fff' }}>
            Candidate Evaluation Pool & Feasibility Breakdown
          </h3>
          <span className="badge badge-indigo">
            {candidates.length} Candidate Models Evaluated
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
            <thead>
              <tr style={{ backgroundColor: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <th style={{ padding: '10px 14px', color: '#94a3b8' }}>Candidate</th>
                <th style={{ padding: '10px 14px', color: '#94a3b8' }}>Strategy</th>
                <th style={{ padding: '10px 14px', color: '#94a3b8' }}>Quality</th>
                <th style={{ padding: '10px 14px', color: '#94a3b8' }}>p50 Latency</th>
                <th style={{ padding: '10px 14px', color: '#94a3b8' }}>Disk Size</th>
                <th style={{ padding: '10px 14px', color: '#94a3b8' }}>Compression</th>
                <th style={{ padding: '10px 14px', color: '#94a3b8' }}>Pareto Optimal</th>
                <th style={{ padding: '10px 14px', color: '#94a3b8' }}>Feasibility Status</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((cand) => {
                const isTop = topCandidate?.candidate_id === cand.candidate_id;
                return (
                  <tr
                    key={cand.candidate_id}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                      backgroundColor: isTop ? 'rgba(16, 185, 129, 0.08)' : 'transparent'
                    }}
                  >
                    <td style={{ padding: '10px 14px', fontWeight: isTop ? 700 : 500, color: '#fff' }}>
                      {cand.name}
                      {isTop && <span className="badge badge-green" style={{ marginLeft: '6px', fontSize: '9px' }}>Top</span>}
                    </td>
                    <td style={{ padding: '10px 14px', color: '#94a3b8' }}>{cand.strategy}</td>
                    <td className="font-mono" style={{ padding: '10px 14px', color: '#38bdf8', fontWeight: 600 }}>
                      {cand.quality_score.toFixed(4)}
                    </td>
                    <td className="font-mono" style={{ padding: '10px 14px', color: '#22d3ee' }}>
                      {cand.latency_p50_ms.toFixed(3)} ms
                    </td>
                    <td className="font-mono" style={{ padding: '10px 14px', color: '#34d399' }}>
                      {cand.disk_size_kb.toFixed(1)} KB
                    </td>
                    <td className="font-mono" style={{ padding: '10px 14px', color: '#a78bfa' }}>
                      {cand.compression_ratio.toFixed(1)}x
                    </td>
                    <td style={{ padding: '10px 14px' }}>
                      {cand.is_pareto_optimal ? (
                        <span className="badge badge-cyan" style={{ fontSize: '10px' }}>Pareto Frontier</span>
                      ) : (
                        <span style={{ color: '#475569', fontSize: '11px' }}>Dominated</span>
                      )}
                    </td>
                    <td style={{ padding: '10px 14px' }}>
                      {cand.is_feasible ? (
                        <span className="badge badge-green" style={{ fontSize: '10px' }}>Feasible</span>
                      ) : (
                        <span className="badge badge-rose" style={{ fontSize: '10px' }} title={cand.constraint_violations.join('; ')}>
                          Violations ({cand.constraint_violations.length})
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
