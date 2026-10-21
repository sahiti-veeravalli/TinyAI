import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Play,
  Layers,
  ArrowRight,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Sliders,
  RefreshCw,
  Clock,
  HardDrive,
  Target,
  FileCode,
  Info,
  Check,
  ChevronRight
} from 'lucide-react';
import {
  parseObjective,
  launchOptimization,
  fetchHardwarePresets,
  fetchExperiments,
  fetchModels
} from '../api';
import {
  ModelEntry,
  HardwarePreset,
  ParsedObjective,
  ExperimentManifest,
  CandidateEvaluation
} from '../types';
import { WorkspaceTab } from './Header';

interface RunsWorkspaceViewProps {
  activeModel: ModelEntry | null;
  onSelectModel: (model: ModelEntry) => void;
  onNavigateTab: (tab: WorkspaceTab) => void;
  onExperimentComplete: (manifest: ExperimentManifest) => void;
}

export const RunsWorkspaceView: React.FC<RunsWorkspaceViewProps> = ({
  activeModel,
  onSelectModel,
  onNavigateTab,
  onExperimentComplete
}) => {
  // Conversational Prompt State
  const [promptText, setPromptText] = useState<string>(
    'Optimize this model for CPU inference. Keep macro F1 above 0.90, keep the model under 250 KB, and minimize median inference latency.'
  );
  const [interpreting, setInterpreting] = useState<boolean>(false);
  const [parsedResult, setParsedResult] = useState<ParsedObjective | null>(null);

  // Structured Configuration State
  const [presets, setPresets] = useState<HardwarePreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('rpi4');
  const [maxDiskSizeKb, setMaxDiskSizeKb] = useState<number>(250);
  const [maxLatencyMs, setMaxLatencyMs] = useState<number>(1.0);
  const [minQualityScore, setMinQualityScore] = useState<number>(0.85);
  const [priority, setPriority] = useState<string>('balanced');
  const [enabledStrategies, setEnabledStrategies] = useState<string[]>([
    'quantization',
    'pruning',
    'distillation',
    'model_selection'
  ]);

  // Execution State
  const [running, setRunning] = useState<boolean>(false);
  const [currentPhase, setCurrentPhase] = useState<string>('');
  const [progressPct, setProgressPct] = useState<number>(0);
  const [generatedCandidates, setGeneratedCandidates] = useState<CandidateEvaluation[]>([]);
  const [completedManifest, setCompletedManifest] = useState<ExperimentManifest | null>(null);

  // History & Models
  const [allModels, setAllModels] = useState<ModelEntry[]>([]);
  const [pastRuns, setPastRuns] = useState<ExperimentManifest[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    loadHardwarePresetsAndRuns();
  }, []);

  const loadHardwarePresetsAndRuns = async () => {
    try {
      const [presetsList, experimentsList, modelsList] = await Promise.all([
        fetchHardwarePresets(),
        fetchExperiments(),
        fetchModels()
      ]);
      setPresets(presetsList);
      setPastRuns(experimentsList);
      setAllModels(modelsList);
      if (!activeModel && modelsList.length > 0) {
        onSelectModel(modelsList[0]);
      }
    } catch (e: any) {
      console.warn('Initial presets load warning:', e);
    }
  };

  const handleInterpretPrompt = async () => {
    if (!promptText.trim()) return;
    try {
      setInterpreting(true);
      setErrorMsg(null);
      const res = await parseObjective(promptText, {
        max_disk_size_kb: maxDiskSizeKb,
        max_latency_ms: maxLatencyMs,
        min_quality_score: minQualityScore,
        priority: priority
      });
      setParsedResult(res);

      // Sync extracted bounds to structured fields
      if (res.max_disk_size_kb !== undefined) setMaxDiskSizeKb(res.max_disk_size_kb);
      if (res.max_latency_ms !== undefined) setMaxLatencyMs(res.max_latency_ms);
      if (res.min_quality_score !== undefined) setMinQualityScore(res.min_quality_score);
      if (res.priority) setPriority(res.priority);
      if (res.enabled_strategies && res.enabled_strategies.length > 0) {
        setEnabledStrategies(res.enabled_strategies);
      }
      if (res.target_hardware && presets.some(p => p.preset_id === res.target_hardware)) {
        setSelectedPresetId(res.target_hardware);
      }
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to interpret prompt');
    } finally {
      setInterpreting(false);
    }
  };

  const handleApplyPreset = (presetId: string) => {
    setSelectedPresetId(presetId);
    const found = presets.find(p => p.preset_id === presetId);
    if (found) {
      setMaxDiskSizeKb(found.recommended_max_size_kb);
      setMaxLatencyMs(found.recommended_max_latency_ms);
    }
  };

  const toggleStrategy = (strategy: string) => {
    setEnabledStrategies(prev =>
      prev.includes(strategy)
        ? prev.filter(s => s !== strategy)
        : [...prev, strategy]
    );
  };

  const handleLaunchExperiment = async () => {
    if (!activeModel) {
      setErrorMsg('Please select a model artifact before launching optimization.');
      return;
    }

    try {
      setRunning(true);
      setErrorMsg(null);
      setCompletedManifest(null);
      setGeneratedCandidates([]);

      // Multi-phase progress simulation for clear user transparency during backend work
      setCurrentPhase('Ingesting & benchmarking uncompressed baseline...');
      setProgressPct(15);

      const expPromise = launchOptimization({
        dataset_id: activeModel.dataset_id || 'ds_fraud_canonical',
        target_column: 'is_fraud',
        baseline_family: activeModel.model_family || 'lightgbm',
        max_disk_size_kb: maxDiskSizeKb,
        max_latency_ms: maxLatencyMs,
        min_quality_score: minQualityScore,
        priority: priority,
        enabled_strategies: enabledStrategies,
        objective_prompt: promptText
      });

      // Interim phase pulses
      setTimeout(() => {
        setCurrentPhase('Quantizing model weights & leaf thresholds (Float16)...');
        setProgressPct(35);
      }, 700);

      setTimeout(() => {
        setCurrentPhase('Pruning redundant ensemble depth & enforcing L1 sparsity...');
        setProgressPct(55);
      }, 1500);

      setTimeout(() => {
        setCurrentPhase('Distilling soft-target knowledge into student MLP...');
        setProgressPct(75);
      }, 2300);

      setTimeout(() => {
        setCurrentPhase('Calculating non-dominated Pareto frontier & ranking candidates...');
        setProgressPct(90);
      }, 3100);

      const manifest = await expPromise;
      setProgressPct(100);
      setCurrentPhase('Experiment complete! Pareto frontier extracted.');
      setGeneratedCandidates(manifest.candidates);
      setCompletedManifest(manifest);
      setPastRuns(prev => [manifest, ...prev]);
      onExperimentComplete(manifest);
    } catch (e: any) {
      setErrorMsg(e.message || 'Optimization experiment execution failed');
      setCurrentPhase('Experiment failed.');
    } finally {
      setRunning(false);
    }
  };

  const promptPills = [
    { label: 'Raspberry Pi 4 (<1.5ms, <250KB)', prompt: 'Optimize for Raspberry Pi 4 CPU. Keep latency under 1.5ms, model under 250 KB, and preserve F1 above 0.88.' },
    { label: 'Microcontroller (<64KB footprint)', prompt: 'Deploy to ARM Cortex-M bare metal. Max latency 0.8ms, size under 64KB, priority smallest size.' },
    { label: 'High-Throughput Server (<0.5ms SLA)', prompt: 'Optimize for Intel edge gateway. Keep latency under 0.5ms, macro F1 > 0.92, priority lowest latency.' },
    { label: 'Balanced Tabular Deployment', prompt: 'Balanced optimization: preserve F1 score while reducing disk size under 100 KB and latency under 1.0 ms.' }
  ];

  return (
    <div className="animate-fade-in" style={{ padding: '24px 32px 64px', maxWidth: '1240px', margin: '0 auto' }}>
      
      {/* Product Promise & Clear Entry Point */}
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em', marginBottom: '6px' }}>
          Make your model deployment-ready.
        </h1>
        <p style={{ fontSize: '14.5px', color: 'var(--text-secondary)', margin: 0 }}>
          Benchmark optimization strategies against your model's real operational constraints.
        </p>
      </div>

      {/* Model Selection Bar */}
      <div className="card" style={{ padding: '12px 18px', marginBottom: '24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'var(--bg-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Layers size={18} color="var(--accent-primary)" />
          </div>
          <div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Active Model for Optimization</div>
            <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>
              {activeModel ? activeModel.name : 'No model selected'}
              {activeModel && (
                <span className="badge badge-neutral" style={{ marginLeft: '8px', fontSize: '11px' }}>
                  {activeModel.model_family} • {activeModel.disk_size_kb.toFixed(1)} KB
                </span>
              )}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {allModels.length > 1 && (
            <select
              value={activeModel?.model_id || ''}
              onChange={(e) => {
                const found = allModels.find(m => m.model_id === e.target.value);
                if (found) onSelectModel(found);
              }}
              className="select"
              style={{ width: 'auto', fontSize: '12.5px', padding: '6px 10px' }}
            >
              {allModels.map(m => (
                <option key={m.model_id} value={m.model_id}>
                  {m.name} ({m.model_family})
                </option>
              ))}
            </select>
          )}

          <button
            onClick={() => onNavigateTab('models')}
            className="btn btn-secondary btn-sm"
          >
            <span>Manage Models Workspace</span>
            <ChevronRight size={13} />
          </button>
        </div>
      </div>

      {/* Global Alerts */}
      {errorMsg && (
        <div style={{
          backgroundColor: 'var(--accent-danger-subtle)',
          border: '1px solid var(--accent-danger-border)',
          borderRadius: 'var(--radius-md)',
          padding: '12px 16px',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '13px',
          color: 'var(--accent-danger)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle size={16} />
            <span>{errorMsg}</span>
          </div>
          <button onClick={() => setErrorMsg(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 700 }}>✕</button>
        </div>
      )}

      {/* Main Two-Column Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '24px', alignItems: 'start' }}>
        
        {/* Left Column: Conversational Objective + Structured Constraints */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Conversational Objective Prompt Area */}
          <div className="card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sparkles size={15} color="var(--accent-primary)" />
                <span style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                  Conversational Objective
                </span>
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                Natural Language Constraint Parser
              </span>
            </div>

            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
              Describe your target environment, predictive threshold, and latency budget in natural language:
            </p>

            <div className="prompt-container" style={{ padding: '12px 14px', marginBottom: '12px' }}>
              <textarea
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                placeholder="e.g. Optimize this model for CPU inference. Keep macro F1 above 0.90, keep the model under 250 KB, and minimize median inference latency."
                rows={3}
                style={{
                  width: '100%',
                  border: 'none',
                  outline: 'none',
                  resize: 'none',
                  backgroundColor: 'transparent',
                  fontFamily: 'inherit',
                  fontSize: '13.5px',
                  color: 'var(--text-main)',
                  lineHeight: 1.5
                }}
              />
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '8px', paddingTop: '8px', borderTop: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Rule-based deterministic extraction • Never silently relaxes hard constraints
                </div>
                <button
                  onClick={handleInterpretPrompt}
                  disabled={interpreting || !promptText.trim()}
                  className="btn btn-secondary btn-sm"
                  style={{ gap: '5px' }}
                >
                  <Sparkles size={12} className={interpreting ? 'animate-spin' : ''} />
                  <span>{interpreting ? 'Interpreting...' : 'Interpret Objective'}</span>
                </button>
              </div>
            </div>

            {/* Quick Prompt Presets */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '12px' }}>
              {promptPills.map((pill, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setPromptText(pill.prompt);
                    // Trigger immediate parse
                    parseObjective(pill.prompt).then(res => {
                      setParsedResult(res);
                      if (res.max_disk_size_kb !== undefined) setMaxDiskSizeKb(res.max_disk_size_kb);
                      if (res.max_latency_ms !== undefined) setMaxLatencyMs(res.max_latency_ms);
                      if (res.min_quality_score !== undefined) setMinQualityScore(res.min_quality_score);
                      if (res.priority) setPriority(res.priority);
                      if (res.enabled_strategies) setEnabledStrategies(res.enabled_strategies);
                      if (res.target_hardware) setSelectedPresetId(res.target_hardware);
                    }).catch(console.error);
                  }}
                  className="prompt-pill"
                >
                  {pill.label}
                </button>
              ))}
            </div>

            {/* Parser Feedback Banner */}
            {parsedResult && (
              <div style={{
                padding: '10px 14px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: parsedResult.is_contradictory ? 'var(--accent-danger-subtle)' : 'var(--bg-subtle)',
                border: `1px solid ${parsedResult.is_contradictory ? 'var(--accent-danger-border)' : 'var(--border-subtle)'}`,
                fontSize: '12.5px'
              }}>
                <div style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '3px' }}>
                  Parsed Objective Summary:
                </div>
                <div style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: '11.5px', marginBottom: '4px' }}>
                  {parsedResult.interpretation_summary}
                </div>
                {parsedResult.validation_warnings.map((warn, i) => (
                  <div key={i} style={{ color: 'var(--accent-warning)', fontSize: '11.5px', marginTop: '2px' }}>
                    ⚠️ {warn}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Structured Experiment Configuration Editor */}
          <div className="card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sliders size={16} color="var(--text-main)" />
                <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                  Structured Constraints & Priority
                </h3>
              </div>
              <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                Directly editable before launch
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '20px' }}>
              {/* Target Hardware Preset */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Deployment Target Preset
                </label>
                <select
                  value={selectedPresetId}
                  onChange={(e) => handleApplyPreset(e.target.value)}
                  className="select"
                >
                  <option value="generic_cpu">Host CPU (General Purpose)</option>
                  <option value="rpi4">Raspberry Pi 4 (ARM Cortex-A72)</option>
                  <option value="cortex_m">ARM Cortex-M (Microcontroller)</option>
                  <option value="jetson_nano">NVIDIA Jetson Nano (Edge GPU)</option>
                  <option value="intel_nuc">Intel Edge NUC / Industrial Gateway</option>
                </select>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '3px', display: 'block' }}>
                  Pre-fills safe target bounds for physical edge hardware.
                </span>
              </div>

              {/* Optimization Priority */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Optimization Priority
                </label>
                <select
                  value={priority}
                  onChange={(e) => setPriority(e.target.value)}
                  className="select"
                >
                  <option value="balanced">Balanced (Quality + Latency + Size)</option>
                  <option value="lowest_latency">Lowest Latency (Fastest p50 ms)</option>
                  <option value="smallest_size">Smallest Footprint (Max Compression)</option>
                  <option value="highest_quality">Highest Quality (Minimal Degradation)</option>
                </select>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '3px', display: 'block' }}>
                  Determines recommendation ranking among feasible candidates.
                </span>
              </div>

              {/* Latency Budget */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Max Latency SLA (p50 ms)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0.05"
                  max="100.0"
                  value={maxLatencyMs}
                  onChange={(e) => setMaxLatencyMs(parseFloat(e.target.value) || 1.0)}
                  className="input"
                />
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '3px', display: 'block' }}>
                  Hard upper bound for single-sample inference time.
                </span>
              </div>

              {/* Model Size Budget */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Max Model Footprint (KB)
                </label>
                <input
                  type="number"
                  step="10"
                  min="5"
                  max="50000"
                  value={maxDiskSizeKb}
                  onChange={(e) => setMaxDiskSizeKb(parseFloat(e.target.value) || 250)}
                  className="input"
                />
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '3px', display: 'block' }}>
                  Disk and flash storage limit on the target device.
                </span>
              </div>

              {/* Min Quality Score */}
              <div style={{ gridColumn: 'span 2' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Minimum Predictive Quality Threshold ({minQualityScore.toFixed(3)} ROC-AUC / F1)
                  </label>
                  <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                    {(minQualityScore * 100).toFixed(1)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.70"
                  max="0.98"
                  step="0.01"
                  value={minQualityScore}
                  onChange={(e) => setMinQualityScore(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
                />
              </div>
            </div>

            {/* Optimization Strategies Checkboxes */}
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Enabled Compression Strategies
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
                {[
                  { id: 'quantization', name: 'Quantization (Float16 precision)' },
                  { id: 'pruning', name: 'Structural Pruning (Depth & Sparsity)' },
                  { id: 'distillation', name: 'Knowledge Distillation (MLP Student)' },
                  { id: 'model_selection', name: 'Compact Model Selection (Shallow Tree / L1)' }
                ].map((st) => (
                  <label
                    key={st.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '8px 12px',
                      borderRadius: 'var(--radius-sm)',
                      backgroundColor: enabledStrategies.includes(st.id) ? 'var(--bg-subtle)' : 'transparent',
                      border: '1px solid',
                      borderColor: enabledStrategies.includes(st.id) ? 'var(--border-medium)' : 'var(--border-subtle)',
                      fontSize: '12.5px',
                      cursor: 'pointer'
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={enabledStrategies.includes(st.id)}
                      onChange={() => toggleStrategy(st.id)}
                      style={{ accentColor: 'var(--accent-primary)' }}
                    />
                    <span>{st.name}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Primary Action Button */}
            <div style={{ marginTop: '24px', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                onClick={handleLaunchExperiment}
                disabled={running || !activeModel}
                className="btn btn-primary btn-lg"
                style={{ minWidth: '200px', gap: '8px' }}
              >
                {running ? (
                  <>
                    <RefreshCw size={15} className="animate-spin" />
                    <span>Running Optimization...</span>
                  </>
                ) : (
                  <>
                    <Play size={15} fill="currentColor" />
                    <span>Run Optimization</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Real Execution State & Candidate Stream */}
          {(running || completedManifest) && (
            <div className="card" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {running ? (
                    <RefreshCw size={15} className="animate-spin" color="var(--accent-primary)" />
                  ) : (
                    <CheckCircle2 size={16} color="var(--accent-success)" />
                  )}
                  <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-main)' }}>
                    {running ? 'Execution in Progress' : 'Experiment Run Completed'}
                  </span>
                </div>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  {progressPct}%
                </span>
              </div>

              {/* Progress Bar */}
              <div style={{
                height: '6px',
                width: '100%',
                backgroundColor: 'var(--bg-subtle)',
                borderRadius: '3px',
                overflow: 'hidden',
                marginBottom: '12px'
              }}>
                <div style={{
                  height: '100%',
                  width: `${progressPct}%`,
                  backgroundColor: running ? 'var(--accent-primary)' : 'var(--accent-success)',
                  transition: 'width 0.4s ease'
                }} />
              </div>

              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                {currentPhase}
              </div>

              {/* Generated Candidates Quick Feed */}
              {generatedCandidates.length > 0 && (
                <div>
                  <div style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
                    Generated Candidates ({generatedCandidates.length})
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {generatedCandidates.map((cand) => (
                      <div
                        key={cand.candidate_id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '8px 12px',
                          borderRadius: 'var(--radius-sm)',
                          backgroundColor: 'var(--bg-subtle)',
                          fontSize: '12.5px'
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{cand.name}</span>
                          <span className="badge badge-neutral" style={{ fontSize: '10px' }}>{cand.strategy}</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>
                          <span>{cand.disk_size_kb.toFixed(1)} KB</span>
                          <span>{cand.latency_p50_ms.toFixed(3)} ms</span>
                          <span>{cand.quality_score.toFixed(3)} {cand.quality_metric_name}</span>
                          {cand.is_feasible ? (
                            <span className="badge badge-success" style={{ fontSize: '10px' }}>Feasible</span>
                          ) : (
                            <span className="badge badge-danger" style={{ fontSize: '10px' }}>Violates SLA</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Success CTA to Results */}
              {completedManifest && (
                <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    onClick={() => onNavigateTab('results')}
                    className="btn btn-primary"
                    style={{ gap: '6px' }}
                  >
                    <span>View Pareto Frontier & Candidate Comparison</span>
                    <ArrowRight size={14} />
                  </button>
                </div>
              )}
            </div>
          )}

        </div>

        {/* Right Column: Run History & Quick Reopen */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          <div className="card" style={{ padding: '16px' }}>
            <div style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.04em', marginBottom: '12px' }}>
              Past Experiment Runs ({pastRuns.length})
            </div>

            {pastRuns.length === 0 ? (
              <div style={{ padding: '24px 12px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12.5px' }}>
                No past runs recorded. Run your first experiment on the left.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {pastRuns.map((run) => {
                  const rec = run.recommended_candidate;
                  return (
                    <div
                      key={run.experiment_id}
                      onClick={() => {
                        onExperimentComplete(run);
                        onNavigateTab('results');
                      }}
                      style={{
                        padding: '10px 12px',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--border-subtle)',
                        backgroundColor: 'var(--bg-subtle)',
                        cursor: 'pointer',
                        transition: 'all 0.12s ease'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = 'var(--border-medium)';
                        e.currentTarget.style.backgroundColor = 'var(--bg-hover)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = 'var(--border-subtle)';
                        e.currentTarget.style.backgroundColor = 'var(--bg-subtle)';
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '3px' }}>
                        <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                          {run.experiment_id}
                        </span>
                        <span className="badge badge-orange" style={{ fontSize: '10px', padding: '1px 5px' }}>
                          {run.priority}
                        </span>
                      </div>
                      <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                        Baseline: {run.baseline_model_family} • {run.candidates?.length || 0} candidates
                      </div>
                      {rec && (
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Check size={11} color="var(--accent-success)" />
                          <span>Rec: <strong>{rec.name}</strong> ({rec.compression_ratio.toFixed(1)}x)</span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Quick Hardware Targets Reference Card */}
          <div className="card" style={{ padding: '16px' }}>
            <div style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.04em', marginBottom: '10px' }}>
              Host Hardware Reference
            </div>
            <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
              Measurements execute on your host operating CPU using microsecond <code className="font-mono">perf_counter</code> timing loops.
            </div>
            <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '11.5px', color: 'var(--text-muted)' }}>
              <div>• <strong>Single-sample inference</strong> (batch size 1)</div>
              <div>• <strong>Warm-up iterations</strong>: 15 runs</div>
              <div>• <strong>Timing distribution</strong>: 100 sample runs</div>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};
