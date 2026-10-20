import React, { useState, useEffect } from 'react';
import {
  Database,
  ShieldCheck,
  ShieldAlert,
  Layers,
  Sparkles,
  Activity,
  TrendingDown,
  Download,
  Play,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  RefreshCw,
  Sliders,
  Cpu,
  FileCode,
  HardDrive,
  Copy,
  Check,
  Info
} from 'lucide-react';
import {
  fetchDatasets,
  loadSampleDataset,
  validateDataset,
  trainBaseline,
  launchOptimization,
  predictInference,
  getDownloadBundleUrl,
  fetchHardwarePresets,
  fetchModelExplainability
} from '../api';
import {
  DatasetMetadata,
  ValidationReport,
  ExperimentManifest,
  CandidateEvaluation,
  HardwarePreset
} from '../types';

interface ExperimentWorkflowViewProps {
  onNavigateTab: (tab: any) => void;
  onOpenModelDetail?: (modelId: string) => void;
}

export const ExperimentWorkflowView: React.FC<ExperimentWorkflowViewProps> = ({
  onNavigateTab
}) => {
  // Stepper Stage: 1: Data, 2: Schema Audit, 3: Baseline, 4: Hardware Budget, 5: Optimization & Pareto, 6: Export & Live Test
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [highestStepUnlocked, setHighestStepUnlocked] = useState<number>(1);

  // Stage 1: Ingestion
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('ds_fraud_canonical');
  const [activeDatasetMeta, setActiveDatasetMeta] = useState<DatasetMetadata | null>(null);
  const [loadingData, setLoadingData] = useState<boolean>(false);

  // Stage 2: Validation
  const [targetColumn, setTargetColumn] = useState<string>('is_fraud');
  const [taskType, setTaskType] = useState<string>('classification');
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [validating, setValidating] = useState<boolean>(false);

  // Stage 3: Baseline
  const [baselineFamily, setBaselineFamily] = useState<string>('lightgbm');
  const [baselineResult, setBaselineResult] = useState<any | null>(null);
  const [trainingBaseline, setTrainingBaseline] = useState<boolean>(false);

  // Stage 4: Hardware & Optimization Constraints
  const [hardwarePresets, setHardwarePresets] = useState<HardwarePreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('rpi4');
  const [maxDiskSizeKb, setMaxDiskSizeKb] = useState<number>(250);
  const [maxLatencyMs, setMaxLatencyMs] = useState<number>(1.2);
  const [minQualityScore, setMinQualityScore] = useState<number>(0.85);
  const [priority, setPriority] = useState<string>('balanced');
  const [enabledStrategies, setEnabledStrategies] = useState<string[]>([
    'quantization',
    'pruning',
    'distillation',
    'model_selection'
  ]);

  // Stage 5 & 6: Optimization & Pareto Manifest
  const [manifest, setManifest] = useState<ExperimentManifest | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateEvaluation | null>(null);
  const [runningOptimization, setRunningOptimization] = useState<boolean>(false);
  const [copiedSha, setCopiedSha] = useState<boolean>(false);

  // Stage 7: Sandbox live prediction
  const [testPayload, setTestPayload] = useState<Record<string, any>>({
    amt: 45.2,
    lat: 34.05,
    long: -118.25,
    city_pop: 50000,
    unix_time: 1325376000,
    merch_lat: 34.06,
    merch_long: -118.26
  });
  const [predictionResponse, setPredictionResponse] = useState<any | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);

  // Global feedback / errors
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Initialize on mount
  useEffect(() => {
    initWorkflow();
  }, []);

  const initWorkflow = async () => {
    try {
      const [dsList, presets] = await Promise.all([
        fetchDatasets(),
        fetchHardwarePresets()
      ]);
      setDatasets(dsList);
      setHardwarePresets(presets);

      if (dsList.length > 0) {
        const canonical = dsList.find(d => d.dataset_id === 'ds_fraud_canonical') || dsList[0];
        setSelectedDatasetId(canonical.dataset_id);
        setActiveDatasetMeta(canonical);
      }
    } catch (e: any) {
      console.warn('Backend initializing, loading canonical dataset...', e);
    }
  };

  const unlockStep = (step: number) => {
    if (step > highestStepUnlocked) {
      setHighestStepUnlocked(step);
    }
    setCurrentStep(step);
  };

  // 1. Handle sample dataset loading
  const handleLoadSample = async (sampleId: string) => {
    try {
      setLoadingData(true);
      setErrorMsg(null);
      const res = await loadSampleDataset(sampleId);
      setSelectedDatasetId(res.dataset_id);
      setActiveDatasetMeta(res.metadata);
      setDatasets(prev => [res.metadata, ...prev.filter(d => d.dataset_id !== res.dataset_id)]);
      setSuccessMsg(`Ingested ${res.metadata.name} (${res.metadata.row_count} rows, ${res.metadata.column_count} features).`);
      unlockStep(2);
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to load sample dataset');
    } finally {
      setLoadingData(false);
    }
  };

  // 2. Handle validation
  const handleRunValidation = async () => {
    if (!selectedDatasetId) {
      setErrorMsg('Please select a dataset first.');
      return;
    }
    try {
      setValidating(true);
      setErrorMsg(null);
      const report = await validateDataset(selectedDatasetId, targetColumn);
      setValidationReport(report);
      setSuccessMsg('Dataset audit complete. Schema and distribution verified.');
      unlockStep(3);
    } catch (e: any) {
      setErrorMsg(e.message || 'Validation failed');
    } finally {
      setValidating(false);
    }
  };

  // 3. Handle baseline training
  const handleTrainBaseline = async () => {
    if (!selectedDatasetId) return;
    try {
      setTrainingBaseline(true);
      setErrorMsg(null);
      const res = await trainBaseline(selectedDatasetId, targetColumn, baselineFamily);
      setBaselineResult(res);
      setSuccessMsg(`Baseline ${baselineFamily.toUpperCase()} trained. Measured p50: ${res.benchmark.latency_p50_ms.toFixed(3)} ms.`);
      unlockStep(4);
    } catch (e: any) {
      setErrorMsg(e.message || 'Baseline training failed');
    } finally {
      setTrainingBaseline(false);
    }
  };

  // 4. Handle Hardware Preset Choice
  const handleSelectPreset = (preset: HardwarePreset) => {
    setSelectedPresetId(preset.preset_id);
    setMaxDiskSizeKb(preset.recommended_max_size_kb);
    setMaxLatencyMs(preset.recommended_max_latency_ms);
  };

  const toggleStrategy = (strategy: string) => {
    setEnabledStrategies(prev =>
      prev.includes(strategy)
        ? prev.filter(s => s !== strategy)
        : [...prev, strategy]
    );
  };

  // 5. Launch Optimization & Pareto Search
  const handleLaunchOptimization = async () => {
    if (!selectedDatasetId) return;
    try {
      setRunningOptimization(true);
      setErrorMsg(null);
      const exp = await launchOptimization({
        dataset_id: selectedDatasetId,
        target_column: targetColumn,
        baseline_family: baselineFamily,
        max_disk_size_kb: maxDiskSizeKb,
        max_latency_ms: maxLatencyMs,
        min_quality_score: minQualityScore,
        priority: priority,
        enabled_strategies: enabledStrategies
      });
      setManifest(exp);
      const rec = exp.recommended_candidate || exp.candidates[0];
      setSelectedCandidate(rec);
      setSuccessMsg(`Optimization complete! Evaluated ${exp.candidates.length} candidates on host CPU.`);
      unlockStep(5);
    } catch (e: any) {
      setErrorMsg(e.message || 'Optimization launch failed');
    } finally {
      setRunningOptimization(false);
    }
  };

  // 6. Test Live Prediction
  const handleRunTestPrediction = async () => {
    try {
      setPredicting(true);
      setErrorMsg(null);
      const res = await predictInference(testPayload, selectedCandidate?.candidate_id || 'mod_canonical_pruned');
      setPredictionResponse(res);
    } catch (e: any) {
      setErrorMsg(e.message || 'Live inference failed');
    } finally {
      setPredicting(false);
    }
  };

  const copyChecksum = (checksum: string) => {
    navigator.clipboard.writeText(checksum);
    setCopiedSha(true);
    setTimeout(() => setCopiedSha(false), 2000);
  };

  // Steps configuration
  const steps = [
    { num: 1, title: 'Dataset Ingestion', desc: 'CSV/Parquet Upload & Fingerprint' },
    { num: 2, title: 'Schema & Leakage', desc: 'Pre-flight Statistical Audit' },
    { num: 3, title: 'Baseline Model', desc: 'Train & Hardware Micro-benchmark' },
    { num: 4, title: 'Budget Constraints', desc: 'Target Hardware & Latency SLA' },
    { num: 5, title: 'Pareto Optimization', desc: 'Multi-Objective Search' },
    { num: 6, title: 'Export & Sandbox', desc: 'SHA-256 Bundle & Live Test' }
  ];

  return (
    <div className="animate-fade-in" style={{ backgroundColor: '#f8fafc', minHeight: 'calc(100vh - 64px)', padding: '28px 36px 64px' }}>
      <div style={{ maxWidth: '1280px', margin: '0 auto' }}>
        {/* Header Title with Crisp Light Styling */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span style={{
                backgroundColor: '#e5322d',
                color: '#ffffff',
                padding: '3px 8px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: 800,
                letterSpacing: '0.04em'
              }}>
                CORE EXPERIMENT STUDIO
              </span>
              <h1 style={{ fontSize: '24px', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.02em', margin: 0 }}>
                End-to-End Edge Optimization Pipeline
              </h1>
            </div>
            <p style={{ fontSize: '13.5px', color: '#64748b', margin: 0 }}>
              Guided multi-objective workflow: Ingest data, audit leakage, benchmark baselines, compress with hardware bounds, and export verified bundles.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={() => handleLoadSample('fraud_detection_benchmark')}
              disabled={loadingData}
              className="btn btn-secondary btn-sm"
            >
              <RefreshCw size={13} className={loadingData ? 'animate-spin' : ''} />
              <span>Reset to Fraud Benchmark</span>
            </button>
            <button
              onClick={() => onNavigateTab('optimization_lab')}
              className="btn btn-outline btn-sm"
            >
              <TrendingDown size={14} color="#4f46e5" />
              <span>Full Pareto Lab</span>
            </button>
          </div>
        </div>

        {/* Global Alerts */}
        {errorMsg && (
          <div style={{
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '12px',
            padding: '12px 18px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '13px',
            color: '#dc2626'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={16} />
              <span><strong>Error:</strong> {errorMsg}</span>
            </div>
            <button onClick={() => setErrorMsg(null)} style={{ background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer', fontWeight: 700 }}>✕</button>
          </div>
        )}

        {successMsg && (
          <div style={{
            backgroundColor: '#f0fdf4',
            border: '1px solid #bbf7d0',
            borderRadius: '12px',
            padding: '12px 18px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '13px',
            color: '#15803d'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle2 size={16} />
              <span>{successMsg}</span>
            </div>
            <button onClick={() => setSuccessMsg(null)} style={{ background: 'none', border: 'none', color: '#15803d', cursor: 'pointer', fontWeight: 700 }}>✕</button>
          </div>
        )}

        {/* Pipeline Stepper Navigation */}
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          border: '1px solid #e2e8f0',
          padding: '16px 20px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          marginBottom: '28px',
          display: 'grid',
          gridTemplateColumns: 'repeat(6, 1fr)',
          gap: '8px'
        }}>
          {steps.map((st) => {
            const isActive = currentStep === st.num;
            const isDone = st.num < currentStep || (st.num === 1 && activeDatasetMeta) || (st.num === 2 && validationReport) || (st.num === 3 && baselineResult) || (st.num === 5 && manifest);
            const isClickable = st.num <= highestStepUnlocked;

            return (
              <button
                key={st.num}
                onClick={() => isClickable && setCurrentStep(st.num)}
                disabled={!isClickable}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '10px 12px',
                  borderRadius: '10px',
                  border: 'none',
                  backgroundColor: isActive ? '#fef2f2' : isDone ? '#f8fafc' : 'transparent',
                  cursor: isClickable ? 'pointer' : 'not-allowed',
                  opacity: isClickable ? 1 : 0.45,
                  textAlign: 'left',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 800,
                  fontSize: '12px',
                  flexShrink: 0,
                  backgroundColor: isActive ? '#e5322d' : isDone ? '#059669' : '#e2e8f0',
                  color: isActive || isDone ? '#ffffff' : '#64748b'
                }}>
                  {isDone && !isActive ? '✓' : st.num}
                </div>
                <div style={{ overflow: 'hidden' }}>
                  <div style={{
                    fontSize: '12.5px',
                    fontWeight: 700,
                    color: isActive ? '#e5322d' : '#0f172a',
                    whiteSpace: 'nowrap',
                    textOverflow: 'ellipsis',
                    overflow: 'hidden'
                  }}>
                    {st.title}
                  </div>
                  <div style={{ fontSize: '10.5px', color: '#64748b', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>
                    {st.desc}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* STEP 1: DATASET INGESTION */}
        {currentStep === 1 && (
          <div className="card" style={{ padding: '28px 32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '10px', backgroundColor: '#fef2f2', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Database size={20} color="#e5322d" />
              </div>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: 0 }}>
                  Stage 1: Dataset Ingestion & Cryptographic Provenance
                </h2>
                <p style={{ fontSize: '13px', color: '#64748b', margin: 0 }}>
                  Select a canonical tabular benchmark or inspect loaded datasets with SHA-256 fingerprinting.
                </p>
              </div>
            </div>

            {/* Quick Presets */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '24px' }}>
              <div
                onClick={() => handleLoadSample('fraud_detection_benchmark')}
                style={{
                  border: selectedDatasetId === 'ds_fraud_canonical' ? '2px solid #e5322d' : '1px solid #e2e8f0',
                  backgroundColor: selectedDatasetId === 'ds_fraud_canonical' ? '#fffaf9' : '#ffffff',
                  borderRadius: '12px',
                  padding: '18px',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a', margin: 0 }}>
                    Financial POS Fraud Benchmark
                  </h3>
                  <span className="badge badge-red" style={{ fontSize: '10px' }}>10,000 Rows</span>
                </div>
                <p style={{ fontSize: '12.5px', color: '#64748b', margin: '0 0 12px' }}>
                  Sub-millisecond tabular classifier with 18 fraud risk features and extreme class imbalance (1.5% fraud).
                </p>
                <div style={{ fontSize: '11px', color: '#94a3b8', fontFamily: 'monospace' }}>
                  Target: <code>is_fraud</code> | Metric: PR-AUC
                </div>
              </div>

              <div
                onClick={() => handleLoadSample('iot_vibration_anomaly')}
                style={{
                  border: selectedDatasetId.includes('iot') ? '2px solid #e5322d' : '1px solid #e2e8f0',
                  backgroundColor: selectedDatasetId.includes('iot') ? '#fffaf9' : '#ffffff',
                  borderRadius: '12px',
                  padding: '18px',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a', margin: 0 }}>
                    Industrial IoT Vibration Anomaly
                  </h3>
                  <span className="badge badge-amber" style={{ fontSize: '10px' }}>5,000 Rows</span>
                </div>
                <p style={{ fontSize: '12.5px', color: '#64748b', margin: '0 0 12px' }}>
                  On-sensor predictive maintenance for microcontrollers under 50KB RAM and 12 continuous sensor axes.
                </p>
                <div style={{ fontSize: '11px', color: '#94a3b8', fontFamily: 'monospace' }}>
                  Target: <code>anomaly</code> | Metric: F1-Score
                </div>
              </div>
            </div>

            {/* Ingested Metadata Card */}
            {activeDatasetMeta && (
              <div style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                padding: '20px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <span style={{ fontSize: '12px', fontWeight: 800, color: '#475569', textTransform: 'uppercase' }}>
                    Active Ingested Dataset Summary
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: '#64748b' }}>
                    <ShieldCheck size={14} color="#059669" />
                    <span>Fingerprint: <code style={{ color: '#0f172a' }}>{activeDatasetMeta.fingerprint.slice(0, 16)}...</code></span>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '16px' }}>
                  <div>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>Sample Count</span>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a' }}>
                      {activeDatasetMeta.row_count.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>Total Columns</span>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a' }}>
                      {activeDatasetMeta.column_count} features
                    </div>
                  </div>
                  <div>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>Numeric Columns</span>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a' }}>
                      {activeDatasetMeta.numeric_columns.length}
                    </div>
                  </div>
                  <div>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>Memory Footprint</span>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a' }}>
                      {(activeDatasetMeta.memory_bytes / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>

                {/* Column preview chips */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {activeDatasetMeta.all_columns.map((c) => (
                    <span
                      key={c}
                      style={{
                        fontSize: '11px',
                        padding: '2px 8px',
                        borderRadius: '6px',
                        backgroundColor: c === targetColumn ? '#fee2e2' : '#ffffff',
                        color: c === targetColumn ? '#dc2626' : '#475569',
                        border: `1px solid ${c === targetColumn ? '#fca5a5' : '#e2e8f0'}`,
                        fontWeight: c === targetColumn ? 700 : 500
                      }}
                    >
                      {c === targetColumn ? `🎯 ${c} (Target)` : c}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                onClick={() => unlockStep(2)}
                disabled={!activeDatasetMeta}
                className="btn btn-primary"
              >
                <span>Proceed to Schema & Leakage Audit</span>
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        )}

        {/* STEP 2: SCHEMA & LEAKAGE VALIDATION */}
        {currentStep === 2 && (
          <div className="card" style={{ padding: '28px 32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '10px', backgroundColor: '#fff1f2', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <ShieldAlert size={20} color="#e11d48" />
              </div>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: 0 }}>
                  Stage 2: Schema Integrity & Target Leakage Audit
                </h2>
                <p style={{ fontSize: '13px', color: '#64748b', margin: 0 }}>
                  Statistical pre-flight audit for missingness, class imbalance, and target correlation leaks.
                </p>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 700, color: '#334155', marginBottom: '6px' }}>
                  Select Target Prediction Column
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '9px 12px',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '13.5px',
                    backgroundColor: '#ffffff'
                  }}
                >
                  {activeDatasetMeta?.all_columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 700, color: '#334155', marginBottom: '6px' }}>
                  Modeling Task Objective
                </label>
                <select
                  value={taskType}
                  onChange={(e) => setTaskType(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '9px 12px',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '13.5px',
                    backgroundColor: '#ffffff'
                  }}
                >
                  <option value="classification">Binary / Multi-Class Classification</option>
                  <option value="regression">Continuous Value Regression</option>
                </select>
              </div>
            </div>

            <div style={{ marginBottom: '24px' }}>
              <button
                onClick={handleRunValidation}
                disabled={validating}
                className="btn btn-secondary"
              >
                <Activity size={15} className={validating ? 'animate-spin' : ''} />
                <span>{validating ? 'Analyzing Distributions...' : 'Run Statistical Validation Check'}</span>
              </button>
            </div>

            {/* Validation Findings */}
            {validationReport && (
              <div style={{
                backgroundColor: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                padding: '20px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className={`badge ${validationReport.is_valid ? 'badge-green' : 'badge-red'}`}>
                      {validationReport.is_valid ? 'VALIDATED FOR TRAINING' : 'BLOCKERS FOUND'}
                    </span>
                    <span style={{ fontSize: '12.5px', color: '#64748b' }}>
                      Audit Target: <code>{validationReport.target_column}</code>
                    </span>
                  </div>
                  {validationReport.imbalance_ratio && (
                    <span style={{ fontSize: '12px', color: '#b45309', fontWeight: 600 }}>
                      Class Imbalance: 1 : {validationReport.imbalance_ratio.toFixed(1)}
                    </span>
                  )}
                </div>

                {/* Audit Items */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {validationReport.issues.length === 0 ? (
                    <div style={{ fontSize: '13px', color: '#059669', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <CheckCircle2 size={16} /> Zero data quality or leakage issues detected.
                    </div>
                  ) : (
                    validationReport.issues.map((iss, idx) => (
                      <div
                        key={idx}
                        style={{
                          padding: '10px 14px',
                          borderRadius: '8px',
                          backgroundColor: iss.severity === 'ERROR' ? '#fef2f2' : iss.severity === 'WARNING' ? '#fffbeb' : '#f8fafc',
                          border: `1px solid ${iss.severity === 'ERROR' ? '#fecaca' : iss.severity === 'WARNING' ? '#fde68a' : '#e2e8f0'}`,
                          fontSize: '12.5px'
                        }}
                      >
                        <div style={{ fontWeight: 700, color: iss.severity === 'ERROR' ? '#b91c1c' : '#92400e', marginBottom: '2px' }}>
                          [{iss.severity}] {iss.message}
                        </div>
                        <div style={{ color: '#64748b' }}>
                          💡 Recommendation: {iss.recommendation}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <button onClick={() => setCurrentStep(1)} className="btn btn-secondary">
                Back to Ingestion
              </button>
              <button
                onClick={() => unlockStep(3)}
                disabled={!validationReport}
                className="btn btn-primary"
              >
                <span>Proceed to Baseline Training</span>
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        )}

        {/* STEP 3: BASELINE TRAINING & BENCHMARKING */}
        {currentStep === 3 && (
          <div className="card" style={{ padding: '28px 32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '10px', backgroundColor: '#eff6ff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Layers size={20} color="#2563eb" />
              </div>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: 0 }}>
                  Stage 3: Train & Empirically Benchmark Baseline Model
                </h2>
                <p style={{ fontSize: '13px', color: '#64748b', margin: 0 }}>
                  Fit an uncompressed reference model. Microsecond latency and memory RSS are timed directly on your CPU.
                </p>
              </div>
            </div>

            {/* Model Family Selection */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
              {[
                { id: 'lightgbm', title: 'LightGBM Ensemble', desc: 'Gradient boosted trees with leaf-wise splitting. Top predictive quality.', rec: true },
                { id: 'random_forest', title: 'Random Forest', desc: 'Bagged decision tree forest. Highly robust but larger disk footprint.' },
                { id: 'decision_tree', title: 'Single Decision Tree', desc: 'Minimal latency tree. Simple explainability, moderate quality.' }
              ].map((m) => (
                <div
                  key={m.id}
                  onClick={() => setBaselineFamily(m.id)}
                  style={{
                    border: baselineFamily === m.id ? '2px solid #2563eb' : '1px solid #e2e8f0',
                    backgroundColor: baselineFamily === m.id ? '#f0f7ff' : '#ffffff',
                    borderRadius: '12px',
                    padding: '16px',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>{m.title}</span>
                    {m.rec && <span className="badge badge-indigo" style={{ fontSize: '10px' }}>Recommended</span>}
                  </div>
                  <p style={{ fontSize: '12px', color: '#64748b', margin: 0 }}>{m.desc}</p>
                </div>
              ))}
            </div>

            <div style={{ marginBottom: '24px' }}>
              <button
                onClick={handleTrainBaseline}
                disabled={trainingBaseline}
                className="btn btn-primary"
              >
                <Activity size={15} className={trainingBaseline ? 'animate-spin' : ''} />
                <span>{trainingBaseline ? 'Training & Benchmarking on Host CPU...' : `Train ${baselineFamily.toUpperCase()} Baseline`}</span>
              </button>
            </div>

            {/* Baseline Benchmark Results */}
            {baselineResult && (
              <div style={{
                backgroundColor: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                padding: '20px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 800, color: '#0f172a' }}>
                    Baseline Empirical Micro-Benchmark (Host Hardware)
                  </span>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>
                    Trained in {baselineResult.training_time_sec.toFixed(3)}s
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                  <div style={{ padding: '14px', borderRadius: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                    <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', fontWeight: 600 }}>Quality (PR-AUC)</span>
                    <div style={{ fontSize: '20px', fontWeight: 900, color: '#0f172a', margin: '4px 0' }}>
                      {(baselineResult.test_metrics.pr_auc || baselineResult.test_metrics.accuracy || 0).toFixed(4)}
                    </div>
                    <span style={{ fontSize: '11px', color: '#10b981' }}>ROC-AUC: {(baselineResult.test_metrics.roc_auc || 0).toFixed(3)}</span>
                  </div>

                  <div style={{ padding: '14px', borderRadius: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                    <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', fontWeight: 600 }}>Measured p50 Latency</span>
                    <div style={{ fontSize: '20px', fontWeight: 900, color: '#2563eb', margin: '4px 0' }}>
                      {baselineResult.benchmark.latency_p50_ms.toFixed(3)} ms
                    </div>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>p95: {baselineResult.benchmark.latency_p95_ms.toFixed(3)} ms</span>
                  </div>

                  <div style={{ padding: '14px', borderRadius: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                    <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', fontWeight: 600 }}>Artifact Size on Disk</span>
                    <div style={{ fontSize: '20px', fontWeight: 900, color: '#0f172a', margin: '4px 0' }}>
                      {baselineResult.benchmark.disk_size_kb.toFixed(1)} KB
                    </div>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>Serialized binary</span>
                  </div>

                  <div style={{ padding: '14px', borderRadius: '8px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                    <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', fontWeight: 600 }}>Process RSS Memory</span>
                    <div style={{ fontSize: '20px', fontWeight: 900, color: '#0f172a', margin: '4px 0' }}>
                      {baselineResult.benchmark.memory_rss_mb.toFixed(1)} MB
                    </div>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>Peak inference RAM</span>
                  </div>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <button onClick={() => setCurrentStep(2)} className="btn btn-secondary">
                Back to Validation
              </button>
              <button
                onClick={() => unlockStep(4)}
                disabled={!baselineResult}
                className="btn btn-primary"
              >
                <span>Proceed to Budget Constraints</span>
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        )}

        {/* STEP 4: HARDWARE TARGET & BUDGET CONSTRAINTS */}
        {currentStep === 4 && (
          <div className="card" style={{ padding: '28px 32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '10px', backgroundColor: '#f5f3ff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Cpu size={20} color="#7c3aed" />
              </div>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: 0 }}>
                  Stage 4: Edge Hardware Presets & Budget Constraints
                </h2>
                <p style={{ fontSize: '13px', color: '#64748b', margin: 0 }}>
                  Define hard physical limits for edge deployment. Any candidate violating these bounds will be marked infeasible.
                </p>
              </div>
            </div>

            {/* Hardware Target Presets */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '24px' }}>
              {hardwarePresets.map((p) => {
                const isSel = selectedPresetId === p.preset_id;
                return (
                  <div
                    key={p.preset_id}
                    onClick={() => handleSelectPreset(p)}
                    style={{
                      padding: '14px',
                      borderRadius: '10px',
                      border: isSel ? '2px solid #7c3aed' : '1px solid #e2e8f0',
                      backgroundColor: isSel ? '#faf5ff' : '#ffffff',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    <div style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>
                      {p.name}
                    </div>
                    <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '10px' }}>
                      {p.target_cpu}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '11px', color: '#475569' }}>
                      <span>Size SLA: &lt; <strong>{p.recommended_max_size_kb} KB</strong></span>
                      <span>Latency SLA: &lt; <strong>{p.recommended_max_latency_ms} ms</strong></span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Sliders & Constraints */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '28px' }}>
              <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#334155' }}>Max Disk Size Budget</span>
                  <span style={{ fontSize: '13px', fontWeight: 800, color: '#0f172a' }}>{maxDiskSizeKb} KB</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="1000"
                  step="10"
                  value={maxDiskSizeKb}
                  onChange={(e) => setMaxDiskSizeKb(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#334155' }}>Max Latency SLA (p50)</span>
                  <span style={{ fontSize: '13px', fontWeight: 800, color: '#0f172a' }}>{maxLatencyMs} ms</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="5.0"
                  step="0.1"
                  value={maxLatencyMs}
                  onChange={(e) => setMaxLatencyMs(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#334155' }}>Min Quality Score</span>
                  <span style={{ fontSize: '13px', fontWeight: 800, color: '#0f172a' }}>{minQualityScore}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="0.99"
                  step="0.01"
                  value={minQualityScore}
                  onChange={(e) => setMinQualityScore(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* Supported Strategies Selection */}
            <div style={{ marginBottom: '28px' }}>
              <span style={{ display: 'block', fontSize: '13px', fontWeight: 700, color: '#0f172a', marginBottom: '10px' }}>
                Supported Compression &amp; Search Strategies:
              </span>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                {[
                  { id: 'quantization', name: 'Float16 Quantization', desc: 'Downscale split thresholds' },
                  { id: 'pruning', name: 'Structural Pruning', desc: 'Trim tree depth & leaf bounds' },
                  { id: 'distillation', name: 'Neural Student Distill', desc: 'Logit transfer to small MLP' },
                  { id: 'model_selection', name: 'Model Family Selection', desc: 'L1 Linear / Decision Tree' }
                ].map((st) => {
                  const checked = enabledStrategies.includes(st.id);
                  return (
                    <div
                      key={st.id}
                      onClick={() => toggleStrategy(st.id)}
                      style={{
                        padding: '12px',
                        borderRadius: '8px',
                        border: checked ? '1px solid #e5322d' : '1px solid #e2e8f0',
                        backgroundColor: checked ? '#fffaf9' : '#ffffff',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '10px'
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {}}
                        style={{ marginTop: '2px', accentColor: '#e5322d' }}
                      />
                      <div>
                        <div style={{ fontSize: '12.5px', fontWeight: 700, color: '#0f172a' }}>{st.name}</div>
                        <div style={{ fontSize: '11px', color: '#64748b' }}>{st.desc}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <button onClick={() => setCurrentStep(3)} className="btn btn-secondary">
                Back to Baseline
              </button>
              <button
                onClick={handleLaunchOptimization}
                disabled={runningOptimization || enabledStrategies.length === 0}
                className="btn btn-primary"
                style={{ backgroundColor: '#e5322d' }}
              >
                <Sparkles size={16} className={runningOptimization ? 'animate-spin' : ''} />
                <span>{runningOptimization ? 'Executing Optimization & Pareto Profiling...' : 'Launch Pareto Optimization →'}</span>
              </button>
            </div>
          </div>
        )}

        {/* STEP 5: PARETO OPTIMIZATION & FRONTIER */}
        {currentStep === 5 && (
          <div className="card" style={{ padding: '28px 32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '10px', backgroundColor: '#eef2ff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <TrendingDown size={20} color="#4f46e5" />
                </div>
                <div>
                  <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: 0 }}>
                    Stage 5: Pareto Frontier & Multi-Objective Evaluation
                  </h2>
                  <p style={{ fontSize: '13px', color: '#64748b', margin: 0 }}>
                    Real empirical evaluations timed on host CPU. Non-dominated candidates form the Pareto frontier.
                  </p>
                </div>
              </div>

              {manifest && (
                <span className="badge badge-indigo">
                  {manifest.candidates.length} Evaluated Candidates
                </span>
              )}
            </div>

            {manifest && (
              <>
                {/* Candidates Evaluation Table */}
                <div style={{ overflowX: 'auto', marginBottom: '24px', border: '1px solid #e2e8f0', borderRadius: '10px' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>Candidate Model</th>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>Strategy</th>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>Quality ({manifest.candidates[0]?.quality_metric_name || 'Score'})</th>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>Latency (p50)</th>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>Disk Size</th>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>Compression</th>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>Status</th>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>SHA-256 Provenance</th>
                        <th style={{ padding: '10px 14px', color: '#64748b' }}>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {manifest.candidates.map((c) => {
                        const isSelected = selectedCandidate?.candidate_id === c.candidate_id;
                        return (
                          <tr
                            key={c.candidate_id}
                            style={{
                              borderBottom: '1px solid #f1f5f9',
                              backgroundColor: isSelected ? '#fffaf9' : 'transparent'
                            }}
                          >
                            <td style={{ padding: '10px 14px', fontWeight: 700, color: '#0f172a' }}>
                              {c.name}
                              {c.candidate_id === manifest.recommended_candidate?.candidate_id && (
                                <span className="badge badge-green" style={{ marginLeft: '6px', fontSize: '9px' }}>Recommended</span>
                              )}
                            </td>
                            <td style={{ padding: '10px 14px', color: '#64748b' }}>{c.strategy}</td>
                            <td className="font-mono" style={{ padding: '10px 14px', fontWeight: 700, color: '#059669' }}>
                              {c.quality_score.toFixed(4)}
                            </td>
                            <td className="font-mono" style={{ padding: '10px 14px', color: '#2563eb' }}>
                              {c.latency_p50_ms.toFixed(3)} ms
                            </td>
                            <td className="font-mono" style={{ padding: '10px 14px', color: '#0f172a' }}>
                              {c.disk_size_kb.toFixed(1)} KB
                            </td>
                            <td className="font-mono" style={{ padding: '10px 14px', color: '#7c3aed', fontWeight: 600 }}>
                              {c.compression_ratio ? `${c.compression_ratio.toFixed(2)}x` : '—'}
                            </td>
                            <td style={{ padding: '10px 14px' }}>
                              <span className={`badge ${c.is_feasible ? (c.is_pareto_optimal ? 'badge-green' : 'badge-cyan') : 'badge-amber'}`}>
                                {c.is_feasible ? (c.is_pareto_optimal ? 'Pareto Optimal' : 'Dominated') : 'Exceeds SLA'}
                              </span>
                            </td>
                            <td className="font-mono" style={{ padding: '10px 14px', color: '#64748b', fontSize: '11px' }}>
                              {c.artifact_checksum_sha256 ? `${c.artifact_checksum_sha256.slice(0, 10)}...` : '—'}
                            </td>
                            <td style={{ padding: '10px 14px' }}>
                              <button
                                onClick={() => setSelectedCandidate(c)}
                                className={`btn btn-sm ${isSelected ? 'btn-primary' : 'btn-outline'}`}
                                style={{ padding: '4px 10px', fontSize: '11px' }}
                              >
                                {isSelected ? 'Selected' : 'Select'}
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Selected Candidate Detail Pill */}
                {selectedCandidate && (
                  <div style={{
                    backgroundColor: '#f8fafc',
                    border: '1px solid #e2e8f0',
                    borderRadius: '12px',
                    padding: '18px 20px',
                    marginBottom: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                  }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        <span style={{ fontSize: '14px', fontWeight: 800, color: '#0f172a' }}>
                          Selected Target: {selectedCandidate.name}
                        </span>
                        <span className="badge badge-indigo">{selectedCandidate.strategy}</span>
                      </div>
                      <p style={{ fontSize: '12px', color: '#64748b', margin: 0 }}>
                        {selectedCandidate.recommendation_rationale || 'Selected for production edge bundle export.'}
                      </p>
                    </div>

                    <button
                      onClick={() => unlockStep(6)}
                      className="btn btn-primary"
                      style={{ backgroundColor: '#e5322d' }}
                    >
                      <span>Proceed to Export & Live Sandbox</span>
                      <ArrowRight size={16} />
                    </button>
                  </div>
                )}
              </>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <button onClick={() => setCurrentStep(4)} className="btn btn-secondary">
                Back to Constraints
              </button>
            </div>
          </div>
        )}

        {/* STEP 6: EXPORT & LIVE SANDBOX */}
        {currentStep === 6 && (
          <div className="card" style={{ padding: '28px 32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '10px', backgroundColor: '#ecfeff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Download size={20} color="#0891b2" />
              </div>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: 0 }}>
                  Stage 6: Cryptographic Artifact Export &amp; Live Prediction Sandbox
                </h2>
                <p style={{ fontSize: '13px', color: '#64748b', margin: 0 }}>
                  Download standalone edge deployment bundle with SHA-256 verification, and test live inference responses.
                </p>
              </div>
            </div>

            {selectedCandidate && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '28px' }}>
                {/* Left: Bundle Download & Cryptographic Provenance */}
                <div style={{
                  padding: '24px',
                  borderRadius: '12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.03)'
                }}>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                    Production Edge Bundle
                  </span>
                  <h3 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: '4px 0 12px' }}>
                    {selectedCandidate.name}
                  </h3>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px', fontSize: '12.5px', color: '#475569' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Model Artifact Size:</span>
                      <strong>{selectedCandidate.disk_size_kb.toFixed(1)} KB</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Measured p50 Latency:</span>
                      <strong style={{ color: '#2563eb' }}>{selectedCandidate.latency_p50_ms.toFixed(3)} ms</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Quality Retention:</span>
                      <strong style={{ color: '#059669' }}>{(selectedCandidate.quality_score * 100).toFixed(1)}% ({selectedCandidate.quality_metric_name})</strong>
                    </div>
                  </div>

                  {/* SHA-256 Provenance Box */}
                  {selectedCandidate.artifact_checksum_sha256 && (
                    <div style={{
                      backgroundColor: '#f8fafc',
                      borderRadius: '8px',
                      padding: '10px 12px',
                      border: '1px solid #e2e8f0',
                      marginBottom: '20px'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontSize: '11px', fontWeight: 700, color: '#475569' }}>SHA-256 Provenance Checksum</span>
                        <button
                          onClick={() => copyChecksum(selectedCandidate.artifact_checksum_sha256 || '')}
                          style={{ background: 'none', border: 'none', color: '#2563eb', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
                        >
                          {copiedSha ? <Check size={12} /> : <Copy size={12} />}
                          <span>{copiedSha ? 'Copied' : 'Copy'}</span>
                        </button>
                      </div>
                      <div className="font-mono" style={{ fontSize: '11px', color: '#0f172a', wordBreak: 'break-all' }}>
                        {selectedCandidate.artifact_checksum_sha256}
                      </div>
                    </div>
                  )}

                  {/* Download Button */}
                  <a
                    href={getDownloadBundleUrl('mod_canonical_pruned')}
                    download
                    className="btn btn-primary"
                    style={{
                      width: '100%',
                      backgroundColor: '#e5322d',
                      padding: '12px 20px',
                      textDecoration: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px'
                    }}
                  >
                    <Download size={18} />
                    <span>Download Standalone Edge Bundle (.zip)</span>
                  </a>
                  <span style={{ display: 'block', textAlign: 'center', fontSize: '11px', color: '#94a3b8', marginTop: '8px' }}>
                    Contains: <code>Dockerfile</code>, <code>serve.py</code>, <code>preprocessor.pkl</code>, <code>model.pkl</code>, <code>manifest.json</code>
                  </span>
                </div>

                {/* Right: Live Inference Sandbox */}
                <div style={{
                  padding: '24px',
                  borderRadius: '12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.03)'
                }}>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                    Hardware Verification
                  </span>
                  <h3 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: '4px 0 12px' }}>
                    Live Inference Testing
                  </h3>
                  <p style={{ fontSize: '12.5px', color: '#64748b', marginBottom: '16px' }}>
                    Send a test payload into the compiled model to measure real-time latency on host hardware.
                  </p>

                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                      Test Feature Payload (JSON)
                    </label>
                    <textarea
                      value={JSON.stringify(testPayload, null, 2)}
                      onChange={(e) => {
                        try {
                          setTestPayload(JSON.parse(e.target.value));
                        } catch {}
                      }}
                      rows={5}
                      className="font-mono"
                      style={{
                        width: '100%',
                        padding: '10px',
                        borderRadius: '8px',
                        border: '1px solid #cbd5e1',
                        fontSize: '11.5px',
                        backgroundColor: '#f8fafc'
                      }}
                    />
                  </div>

                  <button
                    onClick={handleRunTestPrediction}
                    disabled={predicting}
                    className="btn btn-secondary"
                    style={{ width: '100%', marginBottom: '16px' }}
                  >
                    <Play size={15} />
                    <span>{predicting ? 'Executing Prediction...' : 'Send Test Inference Query'}</span>
                  </button>

                  {/* Prediction Response Display */}
                  {predictionResponse && (
                    <div style={{
                      backgroundColor: '#f0fdf4',
                      border: '1px solid #bbf7d0',
                      borderRadius: '8px',
                      padding: '12px 14px',
                      fontSize: '12px'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ fontWeight: 700, color: '#15803d' }}>
                          Prediction Output: {predictionResponse.prediction} ({predictionResponse.prediction === 1 ? 'FRAUD' : 'LEGIT'})
                        </span>
                        <span className="font-mono" style={{ color: '#2563eb', fontWeight: 700 }}>
                          {predictionResponse.latency_ms.toFixed(2)} ms
                        </span>
                      </div>
                      {predictionResponse.probability !== null && (
                        <div style={{ color: '#475569' }}>
                          Probability Score: {(predictionResponse.probability * 100).toFixed(2)}%
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <button onClick={() => setCurrentStep(5)} className="btn btn-secondary">
                Back to Pareto Explorer
              </button>
              <button
                onClick={() => onNavigateTab('monitoring')}
                className="btn btn-outline"
              >
                <span>View Real-Time Drift &amp; Observability →</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
