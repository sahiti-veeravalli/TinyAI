import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Sparkles,
  Download,
  ShieldCheck,
  TrendingDown,
  Layers,
  Clock,
  HardDrive,
  Activity,
  FileCheck,
  Check,
  Info
} from 'lucide-react';
import {
  ExperimentManifest,
  CandidateEvaluation,
  ModelEntry
} from '../types';
import { fetchModelExplainability, exportModelBundle } from '../api';
import { WorkspaceTab } from './Header';

interface ResultsWorkspaceViewProps {
  manifest: ExperimentManifest | null;
  onNavigateTab: (tab: WorkspaceTab) => void;
  onPromoteToRelease: (modelId: string) => void;
}

export const ResultsWorkspaceView: React.FC<ResultsWorkspaceViewProps> = ({
  manifest,
  onNavigateTab,
  onPromoteToRelease
}) => {
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateEvaluation | null>(null);
  const [explainability, setExplainability] = useState<any | null>(null);
  const [loadingExplainability, setLoadingExplainability] = useState<boolean>(false);
  const [exporting, setExporting] = useState<boolean>(false);
  const [exportSuccessMsg, setExportSuccessMsg] = useState<string | null>(null);
  const [exportErrorMsg, setExportErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (manifest && manifest.candidates && manifest.candidates.length > 0) {
      const top = manifest.recommended_candidate || manifest.candidates[0];
      setSelectedCandidate(top);
    }
  }, [manifest]);

  useEffect(() => {
    if (selectedCandidate) {
      loadExplainability(selectedCandidate.candidate_id);
    }
  }, [selectedCandidate]);

  const loadExplainability = async (candidateId: string) => {
    try {
      setLoadingExplainability(true);
      const res = await fetchModelExplainability(candidateId);
      setExplainability(res);
    } catch {
      setExplainability(null);
    } finally {
      setLoadingExplainability(false);
    }
  };

  const handleExportCandidate = async () => {
    if (!selectedCandidate) return;
    try {
      setExporting(true);
      setExportErrorMsg(null);
      setExportSuccessMsg(null);
      await exportModelBundle(selectedCandidate.candidate_id);
      setExportSuccessMsg(`Successfully packaged production deployment bundle for ${selectedCandidate.name}.`);
      onPromoteToRelease(selectedCandidate.candidate_id);
    } catch (e: any) {
      setExportErrorMsg(e.message || 'Export failed');
    } finally {
      setExporting(false);
    }
  };

  if (!manifest) {
    return (
      <div className="animate-fade-in" style={{ padding: '64px 32px', maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
        <BarChart3 size={48} color="var(--text-dim)" style={{ marginBottom: '16px' }} />
        <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', marginBottom: '8px' }}>
          No Experiment Results Yet
        </h2>
        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '24px', lineHeight: 1.5 }}>
          Run an optimization experiment in the Runs workspace to benchmark your model against physical constraints and view the non-dominated Pareto frontier.
        </p>
        <button
          onClick={() => onNavigateTab('runs')}
          className="btn btn-primary btn-lg"
          style={{ gap: '8px' }}
        >
          <span>Go to Runs Workspace</span>
          <ArrowRight size={15} />
        </button>
      </div>
    );
  }

  const candidates = manifest.candidates || [];
  const topCandidate = manifest.recommended_candidate || candidates[0];
  const paretoFrontier = manifest.pareto_frontier || [];
  const allInfeasible = candidates.length > 0 && candidates.every(c => !c.is_feasible);

  // Pareto Chart Data Bounds
  const latencies = candidates.map(c => c.latency_p50_ms);
  const qualities = candidates.map(c => c.quality_score);
  const minLat = Math.min(...latencies, 0.1);
  const maxLat = Math.max(...latencies, 1.5);
  const minQual = Math.min(...qualities, 0.70);
  const maxQual = Math.max(...qualities, 1.0);

  return (
    <div className="animate-fade-in" style={{ padding: '24px 32px 64px', maxWidth: '1240px', margin: '0 auto' }}>
      
      {/* Experiment Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <h1 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-main)', letterSpacing: '-0.02em', margin: 0 }}>
              Optimization Results & Pareto Frontier
            </h1>
            {allInfeasible ? (
              <span className="badge badge-danger">
                <AlertTriangle size={12} />
                <span>Hard Constraints Infeasible</span>
              </span>
            ) : (
              <span className="badge badge-success">
                <CheckCircle2 size={12} />
                <span>Feasible Deployment Solution</span>
              </span>
            )}
          </div>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Experiment ID: <span className="font-mono">{manifest.experiment_id}</span> • Priority: <strong>{manifest.priority.replace('_', ' ').toUpperCase()}</strong> • Evaluated on host AMD64/ARM CPU
          </div>
        </div>

        <button
          onClick={() => onNavigateTab('runs')}
          className="btn btn-secondary btn-sm"
          style={{ gap: '6px' }}
        >
          <span>Modify Constraints / New Run</span>
          <ArrowRight size={13} />
        </button>
      </div>

      {/* Global Alerts */}
      {exportSuccessMsg && (
        <div style={{
          backgroundColor: 'var(--accent-success-subtle)',
          border: '1px solid var(--accent-success-border)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 14px',
          marginBottom: '16px',
          fontSize: '13px',
          color: 'var(--accent-success)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <CheckCircle2 size={15} />
            <span>{exportSuccessMsg}</span>
          </div>
          <button
            onClick={() => onNavigateTab('releases')}
            className="btn btn-primary btn-sm"
            style={{ padding: '3px 8px', fontSize: '11.5px' }}
          >
            <span>View in Releases Workspace</span>
            <ArrowRight size={12} />
          </button>
        </div>
      )}

      {exportErrorMsg && (
        <div style={{
          backgroundColor: 'var(--accent-danger-subtle)',
          border: '1px solid var(--accent-danger-border)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 14px',
          marginBottom: '16px',
          fontSize: '13px',
          color: 'var(--accent-danger)'
        }}>
          {exportErrorMsg}
        </div>
      )}

      {/* Hero Recommendation Rationale Callout Banner */}
      <div style={{
        padding: '16px 20px',
        borderRadius: 'var(--radius-lg)',
        backgroundColor: allInfeasible ? 'var(--accent-warning-subtle)' : 'var(--accent-primary-subtle)',
        border: `1px solid ${allInfeasible ? 'var(--accent-warning-border)' : 'var(--accent-primary-border)'}`,
        marginBottom: '24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <Sparkles size={16} color={allInfeasible ? 'var(--accent-warning)' : 'var(--accent-primary)'} />
          <span style={{ fontSize: '13.5px', fontWeight: 700, color: 'var(--text-main)' }}>
            {allInfeasible ? 'No Candidate Satisfied All Hard Constraints' : 'Selected Candidate Rationale'}
          </span>
        </div>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>
          {topCandidate?.recommendation_rationale || 'Selected based on multi-objective trade-offs.'}
        </p>

        {manifest.constraints && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-subtle)', fontSize: '12px', color: 'var(--text-muted)' }}>
            <span>Target Latency SLA: <strong>&le; {manifest.constraints.max_latency_ms || 'Unbounded'} ms</strong></span>
            <span>Target Disk Budget: <strong>&le; {manifest.constraints.max_disk_size_kb || 'Unbounded'} KB</strong></span>
            <span>Min Quality: <strong>&ge; {manifest.constraints.min_quality_score || 'None'}</strong></span>
            <span>Priority: <strong>{manifest.priority.replace('_', ' ').toUpperCase()}</strong></span>
          </div>
        )}
      </div>

      {/* Main Candidate Comparison Matrix (Table) */}
      <div className="card" style={{ padding: '20px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
              Candidate Comparison Matrix
            </h3>
            <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', margin: '2px 0 0' }}>
              Real hardware measurements captured under single-sample execution (batch size 1).
            </p>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {candidates.length} candidates evaluated
          </div>
        </div>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Candidate Model</th>
                <th>Optimization Strategy</th>
                <th>Predictive Quality</th>
                <th>p50 Latency</th>
                <th>Footprint</th>
                <th>Compression</th>
                <th>Feasibility</th>
                <th style={{ textAlign: 'right' }}>Score</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((cand) => {
                const isSelected = selectedCandidate?.candidate_id === cand.candidate_id;
                const isRecommended = topCandidate?.candidate_id === cand.candidate_id;
                const isBaseline = cand.strategy === 'uncompressed_baseline';

                return (
                  <tr
                    key={cand.candidate_id}
                    onClick={() => setSelectedCandidate(cand)}
                    className={isSelected ? 'row-selected' : ''}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: isRecommended ? 700 : 500, color: 'var(--text-main)' }}>
                          {cand.name}
                        </span>
                        {isRecommended && (
                          <span className="badge badge-orange" style={{ fontSize: '10px', padding: '1px 5px' }}>
                            Top Pick
                          </span>
                        )}
                        {cand.is_pareto_optimal && (
                          <span className="badge badge-info" style={{ fontSize: '10px', padding: '1px 5px' }}>
                            Pareto
                          </span>
                        )}
                      </div>
                    </td>

                    <td>
                      <span className="badge badge-neutral" style={{ fontSize: '11px' }}>
                        {cand.strategy.replace('_', ' ')}
                      </span>
                    </td>

                    <td className="font-mono" style={{ fontWeight: 600 }}>
                      {cand.quality_score ? (
                        <span>
                          {cand.quality_score.toFixed(4)}{' '}
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 400 }}>
                            {cand.quality_metric_name}
                          </span>
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>Not measured</span>
                      )}
                    </td>

                    <td className="font-mono">
                      {cand.latency_p50_ms ? (
                        <span>{cand.latency_p50_ms.toFixed(3)} ms</span>
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>Not measured</span>
                      )}
                    </td>

                    <td className="font-mono">
                      {cand.disk_size_kb ? `${cand.disk_size_kb.toFixed(1)} KB` : 'Not measured'}
                    </td>

                    <td className="font-mono" style={{ fontWeight: isBaseline ? 400 : 600, color: isBaseline ? 'var(--text-muted)' : 'var(--text-main)' }}>
                      {cand.compression_ratio ? `${cand.compression_ratio.toFixed(1)}x` : '1.0x'}
                    </td>

                    <td>
                      {cand.is_feasible ? (
                        <span className="badge badge-success" style={{ gap: '3px' }}>
                          <Check size={11} />
                          <span>Feasible</span>
                        </span>
                      ) : (
                        <span className="badge badge-danger" title={cand.constraint_violations?.join(', ')}>
                          <span>Violates SLA ({cand.constraint_violations?.length || 1})</span>
                        </span>
                      )}
                    </td>

                    <td className="font-mono" style={{ textAlign: 'right', fontWeight: 600 }}>
                      {cand.recommendation_score ? cand.recommendation_score.toFixed(3) : '0.000'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Two-Column Deep Dive: Interactive Pareto Visualizer on Left, Candidate Inspector on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 440px', gap: '24px', alignItems: 'start' }}>
        
        {/* Pareto Frontier Trade-Offs Scatter Visualizer */}
        <div className="card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <div>
              <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                Interactive Pareto Frontier Trade-Offs
              </h3>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '2px 0 0' }}>
                Multi-objective frontier: Quality (higher is better) vs Latency (lower is better).
              </p>
            </div>
            <span className="badge badge-info" style={{ fontSize: '11px' }}>
              {paretoFrontier.length} Non-Dominated Models
            </span>
          </div>

          {/* Clean SVG Scatter Plot */}
          <div style={{
            position: 'relative',
            width: '100%',
            height: '280px',
            backgroundColor: 'var(--bg-subtle)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            padding: '16px 20px 24px 36px',
            overflow: 'hidden'
          }}>
            <svg width="100%" height="100%" viewBox="0 0 500 240" preserveAspectRatio="none">
              {/* Gridlines */}
              <line x1="40" y1="20" x2="480" y2="20" stroke="var(--border-subtle)" strokeDasharray="3 3" />
              <line x1="40" y1="80" x2="480" y2="80" stroke="var(--border-subtle)" strokeDasharray="3 3" />
              <line x1="40" y1="140" x2="480" y2="140" stroke="var(--border-subtle)" strokeDasharray="3 3" />
              <line x1="40" y1="200" x2="480" y2="200" stroke="var(--border-medium)" />
              <line x1="40" y1="20" x2="40" y2="200" stroke="var(--border-medium)" />

              {/* Candidates Points */}
              {candidates.map((cand) => {
                // Latency on X axis: minLat -> maxLat (inverted so lower latency is to the left or right)
                const normX = Math.max(0, Math.min(1, (cand.latency_p50_ms - minLat) / (maxLat - minLat || 1)));
                const normY = Math.max(0, Math.min(1, (cand.quality_score - minQual) / (maxQual - minQual || 1)));
                
                const cx = 50 + normX * 410;
                const cy = 190 - normY * 160;
                
                const isSelected = selectedCandidate?.candidate_id === cand.candidate_id;
                const isRecommended = topCandidate?.candidate_id === cand.candidate_id;

                return (
                  <g
                    key={cand.candidate_id}
                    onClick={() => setSelectedCandidate(cand)}
                    style={{ cursor: 'pointer' }}
                  >
                    {/* Ring for selected */}
                    {isSelected && (
                      <circle cx={cx} cy={cy} r="12" fill="none" stroke="var(--accent-primary)" strokeWidth="2" strokeDasharray="2 2" />
                    )}

                    <circle
                      cx={cx}
                      cy={cy}
                      r={isRecommended ? "7" : "5"}
                      fill={isRecommended ? "var(--accent-primary)" : (cand.is_feasible ? "var(--text-main)" : "var(--accent-danger)")}
                      stroke="var(--bg-surface)"
                      strokeWidth="1.5"
                    />

                    {/* Point label */}
                    <text
                      x={cx + 8}
                      y={cy + 3}
                      fontSize="9.5"
                      fill={isSelected ? "var(--text-main)" : "var(--text-muted)"}
                      fontFamily="var(--font-sans)"
                      fontWeight={isSelected ? "700" : "500"}
                    >
                      {cand.name.split(' ')[0]} ({cand.latency_p50_ms.toFixed(2)}ms)
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Axis Labels */}
            <div style={{ position: 'absolute', bottom: '6px', right: '20px', fontSize: '11px', color: 'var(--text-muted)' }}>
              Inference Latency (p50 ms) &rarr;
            </div>
            <div style={{ position: 'absolute', top: '10px', left: '10px', fontSize: '11px', color: 'var(--text-muted)', transform: 'rotate(-90deg)', transformOrigin: 'left top' }}>
              Quality &uarr;
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '18px', marginTop: '12px', fontSize: '11.5px', color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-primary)' }} />
              <span>Recommended Model</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--text-main)' }} />
              <span>Feasible Candidate</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-danger)' }} />
              <span>Violates SLA</span>
            </div>
          </div>
        </div>

        {/* Selected Candidate Deep Dive Inspector */}
        {selectedCandidate ? (
          <div className="card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '14px' }}>
              <div>
                <span className="badge badge-neutral" style={{ fontSize: '11px', marginBottom: '4px' }}>
                  {selectedCandidate.strategy}
                </span>
                <h3 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                  {selectedCandidate.name}
                </h3>
              </div>

              <button
                onClick={handleExportCandidate}
                disabled={exporting}
                className="btn btn-primary btn-sm"
                style={{ gap: '6px' }}
              >
                <Download size={13} className={exporting ? 'animate-spin' : ''} />
                <span>{exporting ? 'Packaging...' : 'Export Release'}</span>
              </button>
            </div>

            {/* Performance Specs Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '10px',
              padding: '12px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-subtle)',
              marginBottom: '16px'
            }}>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Quality Metric</div>
                <div className="font-mono" style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                  {selectedCandidate.quality_score.toFixed(4)} {selectedCandidate.quality_metric_name}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Median Latency</div>
                <div className="font-mono" style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                  {selectedCandidate.latency_p50_ms.toFixed(3)} ms
                </div>
              </div>

              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Disk Size Footprint</div>
                <div className="font-mono" style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                  {selectedCandidate.disk_size_kb.toFixed(1)} KB
                </div>
              </div>

              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Compression Ratio</div>
                <div className="font-mono" style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                  {selectedCandidate.compression_ratio.toFixed(1)}x
                </div>
              </div>
            </div>

            {/* Constraint Status List */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
                Operational Constraint Evaluation
              </div>
              {selectedCandidate.is_feasible ? (
                <div style={{ fontSize: '12px', color: 'var(--accent-success)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Check size={14} />
                  <span>Strictly satisfies all physical deployment budgets.</span>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {selectedCandidate.constraint_violations?.map((violation, idx) => (
                    <div key={idx} style={{ fontSize: '12px', color: 'var(--accent-danger)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <AlertTriangle size={13} />
                      <span>{violation}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Explainability / Feature Fidelity */}
            <div>
              <div style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
                Explainability & Feature Fidelity
              </div>
              {loadingExplainability ? (
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Analyzing feature importance fidelity...</div>
              ) : explainability ? (
                <div style={{ padding: '10px 12px', borderRadius: 'var(--radius-sm)', backgroundColor: 'var(--bg-subtle)', fontSize: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span>Attribution Cosine Similarity:</span>
                    <strong className="font-mono">{explainability.fidelity_score_pct}%</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Rank Correlation (Spearman):</span>
                    <strong className="font-mono">{explainability.rank_correlation}</strong>
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Explainability metrics available once bundle is packaged.</div>
              )}
            </div>

            {/* Provenance Hash */}
            <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px solid var(--border-subtle)', fontSize: '11.5px', color: 'var(--text-muted)' }}>
              SHA-256: <code className="font-mono" style={{ fontSize: '10.5px' }}>{selectedCandidate.artifact_checksum_sha256 || 'Verified Local Artifact'}</code>
            </div>
          </div>
        ) : null}

      </div>

    </div>
  );
};
