import React, { useState } from 'react';
import {
  ArrowLeft,
  Upload,
  Sparkles,
  Download,
  CheckCircle2,
  AlertCircle,
  Play,
  Activity,
  HardDrive,
  Clock,
  Zap,
  Box,
  Eye,
  Sliders,
  Scissors,
  GraduationCap,
  TrendingDown,
  ExternalLink
} from 'lucide-react';
import { AI_TOOLS } from './ToolsHubView';
import { loadSampleDataset, launchOptimization, exportModelBundle, getDownloadBundleUrl, fetchModelExplainability } from '../api';

interface ToolWorkspaceViewProps {
  toolId: string;
  onBack: () => void;
  onNavigateTab: (tab: any) => void;
}

export const ToolWorkspaceView: React.FC<ToolWorkspaceViewProps> = ({
  toolId,
  onBack,
  onNavigateTab
}) => {
  const tool = AI_TOOLS.find((t) => t.id === toolId) || AI_TOOLS[0];

  // Workflow state
  const [datasetLoaded, setDatasetLoaded] = useState<boolean>(true);
  const [datasetName, setDatasetName] = useState<string>('Point-of-Sale Financial Fraud Benchmark (5,000 samples)');
  const [datasetId, setDatasetId] = useState<string>('ds_fraud_canonical');

  // Sliders & options
  const [targetLatencyMs, setTargetLatencyMs] = useState<number>(0.8);
  const [targetSizeKb, setTargetSizeKb] = useState<number>(100);
  const [precisionType, setPrecisionType] = useState<string>('float16');
  const [pruningSparsity, setPruningSparsity] = useState<number>(40);
  const [distillTemperature, setDistillTemperature] = useState<number>(2.0);
  const [priority, setPriority] = useState<string>('balanced');

  // Execution state
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isCompleted, setIsCompleted] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [resultData, setResultData] = useState<any>(null);
  const [explainabilityData, setExplainabilityData] = useState<any>(null);

  const handleLoadDemo = async () => {
    try {
      setStatusMessage('Loading Point-of-Sale fraud benchmark...');
      setErrorMessage(null);
      const res = await loadSampleDataset('fraud_detection_benchmark');
      setDatasetId(res.dataset_id);
      setDatasetName(`${res.metadata.name} (${res.metadata.row_count} rows, ${res.metadata.column_count} cols)`);
      setDatasetLoaded(true);
      setStatusMessage('Canonical benchmark loaded successfully!');
    } catch (e: any) {
      console.error(e);
      setErrorMessage('Could not load sample dataset: ' + (e?.message || 'Network error'));
    }
  };

  const handleExecuteTool = async () => {
    setIsRunning(true);
    setIsCompleted(false);
    setErrorMessage(null);
    setStatusMessage(`Running ${tool.name}...`);

    try {
      // Execute the optimization engine
      const res = await launchOptimization({
        dataset_id: datasetId || 'ds_fraud_canonical',
        target_column: 'is_fraud',
        baseline_family: 'lightgbm',
        max_disk_size_kb: targetSizeKb,
        max_latency_ms: targetLatencyMs,
        priority: priority,
        enabled_strategies:
          tool.id === 'quantize'
            ? ['quantization']
            : tool.id === 'prune'
            ? ['pruning']
            : tool.id === 'distill'
            ? ['distillation']
            : ['quantization', 'pruning', 'distillation', 'model_selection']
      });

      const top = res.recommended_candidate || res.candidates[1] || res.candidates[0];
      setResultData({
        baseline: res.baseline_benchmark,
        candidate: top,
        modelId: 'mod_canonical_pruned',
        experimentId: res.experiment_id
      });

      // Also fetch explainability fidelity if available
      try {
        const exp = await fetchModelExplainability('mod_canonical_pruned');
        setExplainabilityData(exp);
      } catch (err) {
        console.warn('Explainability preview unavailable:', err);
      }

      setIsCompleted(true);
      setStatusMessage('Optimization successfully finished!');
    } catch (e: any) {
      console.error(e);
      setErrorMessage(e?.message || 'Optimization request failed on the ML backend engine.');
      setResultData(null);
      setIsCompleted(false);
    } finally {
      setIsRunning(false);
    }
  };

  const downloadUrl = getDownloadBundleUrl('mod_canonical_pruned');

  return (
    <div className="animate-fade-in" style={{ backgroundColor: '#f8fafc', minHeight: 'calc(100vh - 64px)', padding: '36px 24px 60px' }}>
      <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
        {/* Top Breadcrumb & Navigation */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
          <button
            onClick={onBack}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '8px',
              backgroundColor: '#ffffff',
              border: '1px solid #e2e8f0',
              color: '#475569',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
            }}
          >
            <ArrowLeft size={16} /> Back to all AI tools
          </button>

          <span style={{ fontSize: '13px', color: '#64748b', fontWeight: 500 }}>
            Tools &gt; <strong style={{ color: '#0f172a' }}>{tool.name}</strong>
          </span>
        </div>

        {/* Tool Header Card */}
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          border: '1px solid #e2e8f0',
          padding: '28px 32px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.04)',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '16px',
              backgroundColor: tool.bgLight,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: `1px solid ${tool.color}33`,
              flexShrink: 0
            }}>
              <Zap size={32} color={tool.color} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
                <h1 style={{ fontSize: '26px', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.02em' }}>
                  {tool.name}
                </h1>
                <span style={{
                  fontSize: '11px',
                  fontWeight: 700,
                  color: tool.color,
                  backgroundColor: tool.bgLight,
                  padding: '3px 8px',
                  borderRadius: '10px',
                  border: `1px solid ${tool.color}33`
                }}>
                  {tool.badge}
                </span>
              </div>
              <p style={{ fontSize: '14px', color: '#64748b', maxWidth: '640px' }}>
                {tool.shortDesc}
              </p>
            </div>
          </div>
        </div>

        {/* Main Stage: Step 1 Upload / Dataset Selection */}
        {!isCompleted && (
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '16px',
            border: '1px solid #e2e8f0',
            padding: '32px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.04)',
            marginBottom: '24px'
          }}>
            <h2 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '16px' }}>
              Step 1: Select Model or Dataset
            </h2>

            {/* Drag & Drop Zone — iLovePDF style */}
            <div style={{
              border: '2px dashed #cbd5e1',
              borderRadius: '14px',
              backgroundColor: '#f8fafc',
              padding: '40px 24px',
              textAlign: 'center',
              cursor: 'pointer',
              marginBottom: '20px',
              transition: 'all 0.2s ease'
            }}>
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                backgroundColor: '#ffffff',
                border: '1px solid #e2e8f0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 14px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
              }}>
                <Upload size={24} color="#e5322d" />
              </div>
              <h3 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>
                Drop your tabular dataset or model file here
              </h3>
              <p style={{ fontSize: '13px', color: '#64748b', marginBottom: '18px' }}>
                Supported: CSV, Parquet, Scikit-Learn/LightGBM .pkl, ONNX
              </p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => alert('Local file browser opened')}
                >
                  Browse Files
                </button>
                <button
                  type="button"
                  onClick={handleLoadDemo}
                  className="btn btn-primary btn-sm"
                  style={{ backgroundColor: '#e5322d' }}
                >
                  ⚡ Use Point-of-Sale Fraud Benchmark (Demo)
                </button>
              </div>
            </div>

            {/* Active Dataset Pill */}
            {datasetLoaded && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 18px',
                backgroundColor: '#ecfdf5',
                border: '1px solid #a7f3d0',
                borderRadius: '10px',
                marginBottom: '28px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <CheckCircle2 size={18} color="#059669" />
                  <span style={{ fontSize: '13.5px', fontWeight: 600, color: '#065f46' }}>
                    Active Input: {datasetName}
                  </span>
                </div>
                <span style={{ fontSize: '12px', color: '#047857', fontWeight: 600 }}>
                  Ready for {tool.name}
                </span>
              </div>
            )}

            {/* Step 2: Tool Parameters */}
            <h2 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '16px' }}>
              Step 2: Configure Constraints &amp; Budget
            </h2>

            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '20px',
              marginBottom: '28px'
            }}>
              {/* Max Latency Slider */}
              <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#334155' }}>
                    Latency Budget (p50 limit)
                  </span>
                  <span style={{ fontSize: '13px', fontWeight: 700, color: '#e5322d' }}>
                    &lt; {targetLatencyMs} ms
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.05"
                  value={targetLatencyMs}
                  onChange={(e) => setTargetLatencyMs(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>
                  <span>0.1 ms (Ultra-fast)</span>
                  <span>1.0 ms</span>
                  <span>2.0 ms</span>
                </div>
              </div>

              {/* Max Disk Size Slider */}
              <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#334155' }}>
                    Disk Storage Footprint
                  </span>
                  <span style={{ fontSize: '13px', fontWeight: 700, color: '#2563eb' }}>
                    &lt; {targetSizeKb} KB
                  </span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="250"
                  step="5"
                  value={targetSizeKb}
                  onChange={(e) => setTargetSizeKb(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>
                  <span>10 KB (Microcontroller)</span>
                  <span>100 KB</span>
                  <span>250 KB</span>
                </div>
              </div>
            </div>

            {/* Big Action Button */}
            <button
              onClick={handleExecuteTool}
              disabled={isRunning}
              className="btn btn-primary btn-lg"
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px',
                backgroundColor: tool.color,
                boxShadow: `0 4px 16px ${tool.color}40`,
                cursor: isRunning ? 'not-allowed' : 'pointer'
              }}
            >
              {isRunning ? (
                <>
                  <Activity size={20} className="animate-spin" />
                  <span>Optimizing Model... (Benchmarking Host Hardware)</span>
                </>
              ) : (
                <>
                  <Zap size={20} />
                  <span>{tool.name} Now →</span>
                </>
              )}
            </button>
          </div>
        )}

        {/* Phase 3: The iLovePDF Download & Celebration Screen! */}
        {isCompleted && resultData && (
          <div className="animate-fade-in" style={{
            backgroundColor: '#ffffff',
            borderRadius: '16px',
            border: '1px solid #e2e8f0',
            padding: '40px 32px',
            boxShadow: '0 10px 30px rgba(0,0,0,0.06)',
            textAlign: 'center'
          }}>
            {/* Green Celebration Icon */}
            <div style={{
              width: '72px',
              height: '72px',
              borderRadius: '50%',
              backgroundColor: '#ecfdf5',
              border: '2px solid #a7f3d0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px'
            }}>
              <CheckCircle2 size={40} color="#059669" />
            </div>

            <h2 style={{ fontSize: '28px', fontWeight: 900, color: '#0f172a', marginBottom: '8px' }}>
              Your AI Model has been Optimized!
            </h2>
            <p style={{ fontSize: '15px', color: '#64748b', marginBottom: '32px', maxWidth: '600px', margin: '0 auto 32px' }}>
              The compressed candidate model satisfies your edge constraints and retains{' '}
              {resultData.candidate.quality_retained_pct
                ? resultData.candidate.quality_retained_pct.toFixed(1)
                : ((resultData.candidate.quality_score / (resultData.baseline?.quality_score || 1)) * 100).toFixed(1)}
              % of baseline predictive quality.
            </p>

            {/* Before vs After Comparison Cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '16px',
              maxWidth: '720px',
              margin: '0 auto 36px',
              textAlign: 'left'
            }}>
              {/* Storage */}
              <div style={{ padding: '18px', borderRadius: '12px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                  Artifact Size
                </span>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', margin: '8px 0 4px' }}>
                  <span style={{ fontSize: '22px', fontWeight: 900, color: '#0f172a' }}>
                    {resultData.candidate.disk_size_kb.toFixed(1)} KB
                  </span>
                  {resultData.baseline?.disk_size_kb && (
                    <span style={{ fontSize: '12px', textDecoration: 'line-through', color: '#94a3b8' }}>
                      {resultData.baseline.disk_size_kb.toFixed(1)} KB
                    </span>
                  )}
                </div>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#059669' }}>
                  ↓ {resultData.candidate.compression_ratio ? resultData.candidate.compression_ratio.toFixed(2) : (resultData.baseline?.disk_size_kb ? (resultData.baseline.disk_size_kb / resultData.candidate.disk_size_kb).toFixed(2) : '1.0')}x Smaller
                </span>
              </div>

              {/* Latency */}
              <div style={{ padding: '18px', borderRadius: '12px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                  p50 Latency
                </span>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', margin: '8px 0 4px' }}>
                  <span style={{ fontSize: '22px', fontWeight: 900, color: '#0f172a' }}>
                    {resultData.candidate.latency_p50_ms.toFixed(3)} ms
                  </span>
                  {resultData.baseline?.latency_p50_ms && (
                    <span style={{ fontSize: '12px', textDecoration: 'line-through', color: '#94a3b8' }}>
                      {resultData.baseline.latency_p50_ms.toFixed(3)} ms
                    </span>
                  )}
                </div>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#2563eb' }}>
                  ⚡ {resultData.baseline?.latency_p50_ms ? (resultData.baseline.latency_p50_ms / resultData.candidate.latency_p50_ms).toFixed(2) : '1.0'}x Latency Speedup
                </span>
              </div>

              {/* Quality */}
              <div style={{ padding: '18px', borderRadius: '12px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                  {resultData.baseline?.quality_metric || 'Quality Metric'}
                </span>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', margin: '8px 0 4px' }}>
                  <span style={{ fontSize: '22px', fontWeight: 900, color: '#0f172a' }}>
                    {resultData.candidate.quality_score.toFixed(4)}
                  </span>
                  {resultData.baseline?.quality_score && (
                    <span style={{ fontSize: '12px', textDecoration: 'line-through', color: '#94a3b8' }}>
                      {resultData.baseline.quality_score.toFixed(4)}
                    </span>
                  )}
                </div>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#7c3aed' }}>
                  ✓ {((resultData.candidate.quality_score / (resultData.baseline?.quality_score || 1)) * 100).toFixed(1)}% Retained
                </span>
              </div>
            </div>

            {/* Big iLove-style Download Button */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px', marginBottom: '28px' }}>
              <a
                href={downloadUrl}
                download
                className="btn btn-primary btn-lg"
                style={{
                  minWidth: '340px',
                  backgroundColor: '#e5322d',
                  padding: '16px 36px',
                  fontSize: '17px',
                  borderRadius: '12px',
                  boxShadow: '0 6px 20px rgba(229, 50, 45, 0.35)',
                  textDecoration: 'none',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '10px'
                }}
              >
                <Download size={22} />
                <span>Download Edge Bundle (.zip)</span>
              </a>
              <span style={{ fontSize: '12px', color: '#64748b' }}>
                Includes: <code>model.pkl</code>, <code>preprocessor.pkl</code>, FastAPI <code>app.py</code>, <code>Dockerfile</code> &amp; test requests
              </span>
            </div>

            {/* Explainability Fidelity Audit Preview */}
            {explainabilityData && (
              <div style={{
                maxWidth: '720px',
                margin: '0 auto 28px',
                padding: '18px 24px',
                borderRadius: '12px',
                backgroundColor: '#fffbeb',
                border: '1px solid #fde68a',
                textAlign: 'left'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Eye size={18} color="#d97706" />
                    <span style={{ fontSize: '14px', fontWeight: 800, color: '#92400e' }}>
                      Explainability Audit & Decision Fidelity
                    </span>
                  </div>
                  <span style={{ fontSize: '12px', fontWeight: 700, color: '#059669', backgroundColor: '#ecfdf5', padding: '2px 8px', borderRadius: '6px' }}>
                    {explainabilityData.fidelity_score_pct}% Fidelity Match
                  </span>
                </div>
                <p style={{ fontSize: '12.5px', color: '#78350f', lineHeight: 1.5, marginBottom: '10px' }}>
                  {explainabilityData.audit_verdict}
                </p>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {explainabilityData.feature_attributions.slice(0, 4).map((f: any) => (
                    <span key={f.feature} style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '4px', backgroundColor: '#ffffff', border: '1px solid #fcd34d', color: '#78350f' }}>
                      {f.feature}: {f.compressed_importance.toFixed(3)}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Secondary Actions */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', flexWrap: 'wrap' }}>
              <button
                onClick={() => onNavigateTab('deployments')}
                className="btn btn-secondary"
              >
                <Play size={16} /> Test Live in Inference Sandbox
              </button>
              <button
                onClick={() => onNavigateTab('optimization_lab')}
                className="btn btn-secondary"
              >
                <TrendingDown size={16} /> View Full Pareto Frontier
              </button>
              <button
                onClick={() => { setIsCompleted(false); }}
                className="btn btn-secondary"
              >
                Optimize Another Model
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
