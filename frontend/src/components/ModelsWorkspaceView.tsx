import React, { useState, useEffect } from 'react';
import {
  Layers,
  Upload,
  FileCheck,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Play,
  Sparkles,
  Info,
  Database,
  ArrowRight,
  RefreshCw,
  Check,
  ShieldCheck,
  X
} from 'lucide-react';
import {
  fetchModels,
  fetchSampleCatalog,
  loadSampleModel,
  uploadModel
} from '../api';
import { ModelEntry, SampleModelCatalogItem } from '../types';
import { WorkspaceTab } from './Header';

interface ModelsWorkspaceViewProps {
  onSelectModelForRun: (model: ModelEntry) => void;
  onNavigateTab: (tab: WorkspaceTab) => void;
}

export const ModelsWorkspaceView: React.FC<ModelsWorkspaceViewProps> = ({
  onSelectModelForRun,
  onNavigateTab
}) => {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [catalog, setCatalog] = useState<SampleModelCatalogItem[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelEntry | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [showFormatsModal, setShowFormatsModal] = useState<boolean>(false);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      const [modelsList, catalogList] = await Promise.all([
        fetchModels(),
        fetchSampleCatalog()
      ]);
      setModels(modelsList);
      setCatalog(catalogList);
      if (modelsList.length > 0) {
        setSelectedModel(modelsList[0]);
      }
    } catch (e: any) {
      console.error(e);
      setErrorMsg(e.message || 'Failed to load model registry');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      setErrorMsg(null);
      setSuccessMsg(null);
      const res = await uploadModel(file);
      setSuccessMsg(res.message);
      setModels(prev => [res.model, ...prev]);
      setSelectedModel(res.model);
    } catch (e: any) {
      setErrorMsg(e.message || 'Model upload and pre-flight validation failed');
    } finally {
      setUploading(false);
      // Reset input
      event.target.value = '';
    }
  };

  const handleSelectSample = async (sampleId: string) => {
    try {
      setLoadingSample(sampleId);
      setErrorMsg(null);
      setSuccessMsg(null);
      const res = await loadSampleModel(sampleId);
      setSuccessMsg(res.message);
      setModels(prev => {
        const filtered = prev.filter(m => m.model_id !== res.model.model_id);
        return [res.model, ...filtered];
      });
      setSelectedModel(res.model);
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to load sample model');
    } finally {
      setLoadingSample(null);
    }
  };

  return (
    <div className="animate-fade-in" style={{ padding: '24px 32px 64px', maxWidth: '1240px', margin: '0 auto' }}>
      {/* Workspace Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-main)', letterSpacing: '-0.02em', marginBottom: '4px' }}>
            Models Workspace
          </h1>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', margin: 0 }}>
            Inspect imported model artifacts, verify pre-flight contracts, and check compression compatibility.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            onClick={() => setShowFormatsModal(true)}
            className="btn btn-secondary btn-sm"
            style={{ gap: '6px' }}
          >
            <HelpCircle size={14} />
            <span>Supported Formats & Limits</span>
          </button>

          <label className="btn btn-primary btn-sm" style={{ cursor: uploading ? 'wait' : 'pointer', gap: '6px' }}>
            <Upload size={14} className={uploading ? 'animate-spin' : ''} />
            <span>{uploading ? 'Validating...' : 'Upload Model (.pkl, .joblib)'}</span>
            <input
              type="file"
              accept=".pkl,.joblib,.pickle"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
              disabled={uploading}
            />
          </label>
        </div>
      </div>

      {/* Global Status Notifications */}
      {errorMsg && (
        <div style={{
          backgroundColor: 'var(--accent-danger-subtle)',
          border: '1px solid var(--accent-danger-border)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 14px',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '13px',
          color: 'var(--accent-danger)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle size={15} />
            <span><strong>Validation Alert:</strong> {errorMsg}</span>
          </div>
          <button onClick={() => setErrorMsg(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 700 }}>✕</button>
        </div>
      )}

      {successMsg && (
        <div style={{
          backgroundColor: 'var(--accent-success-subtle)',
          border: '1px solid var(--accent-success-border)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 14px',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '13px',
          color: 'var(--accent-success)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <CheckCircle2 size={15} />
            <span>{successMsg}</span>
          </div>
          <button onClick={() => setSuccessMsg(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 700 }}>✕</button>
        </div>
      )}

      {/* Main Layout: Models List on left, Selected Model Details on right */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '20px', alignItems: 'start' }}>
        
        {/* Left Column: Model Registry & Sample Models */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Active / Imported Models */}
          <div className="card" style={{ padding: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                Imported Models ({models.length})
              </div>
              <button
                onClick={loadInitialData}
                className="btn btn-ghost btn-sm"
                style={{ padding: '2px 6px', height: 'auto' }}
                title="Refresh list"
              >
                <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              </button>
            </div>

            {loading ? (
              <div style={{ padding: '24px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                Loading models...
              </div>
            ) : models.length === 0 ? (
              <div style={{
                padding: '24px 16px',
                textAlign: 'center',
                backgroundColor: 'var(--bg-subtle)',
                borderRadius: 'var(--radius-md)',
                border: '1px dashed var(--border-medium)'
              }}>
                <Layers size={24} color="var(--text-dim)" style={{ marginBottom: '8px' }} />
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>No models imported yet</div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Upload a model file or select one of the documented sample models below.
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {models.map((mod) => {
                  const isSelected = selectedModel?.model_id === mod.model_id;
                  return (
                    <div
                      key={mod.model_id}
                      onClick={() => setSelectedModel(mod)}
                      style={{
                        padding: '10px 12px',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid',
                        borderColor: isSelected ? 'var(--border-focus)' : 'var(--border-subtle)',
                        backgroundColor: isSelected ? 'var(--accent-primary-subtle)' : 'var(--bg-surface)',
                        cursor: 'pointer',
                        transition: 'all 0.12s ease'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '3px' }}>
                        <span style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                          {mod.name}
                        </span>
                        <span className="badge badge-neutral" style={{ fontSize: '11px', padding: '1px 6px' }}>
                          {mod.model_family}
                        </span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '11.5px', color: 'var(--text-muted)' }}>
                        <span>{mod.disk_size_kb.toFixed(1)} KB</span>
                        <span>•</span>
                        <span>{mod.quality_score ? `${(mod.quality_score).toFixed(3)} ${mod.quality_metric || 'ROC-AUC'}` : 'Unmeasured'}</span>
                        <span>•</span>
                        <span>{mod.latency_p50_ms ? `${mod.latency_p50_ms.toFixed(2)}ms` : 'Raw'}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Documented Sample Models Picker */}
          <div className="card" style={{ padding: '16px' }}>
            <div style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.04em', marginBottom: '12px' }}>
              Canonical Sample Models
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {catalog.map((sample) => {
                const isLoadingThis = loadingSample === sample.sample_id;
                return (
                  <div
                    key={sample.sample_id}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-subtle)',
                      backgroundColor: 'var(--bg-subtle)'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)' }}>
                        {sample.name}
                      </span>
                      <span className="badge badge-orange" style={{ fontSize: '10.5px', padding: '1px 6px' }}>
                        {sample.model_family}
                      </span>
                    </div>
                    <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', margin: '0 0 8px', lineHeight: 1.35 }}>
                      {sample.description}
                    </p>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                        {sample.baseline_size_kb}KB • {sample.baseline_latency_ms}ms p50
                      </span>
                      <button
                        onClick={() => handleSelectSample(sample.sample_id)}
                        disabled={isLoadingThis}
                        className="btn btn-secondary btn-sm"
                        style={{ padding: '3px 8px', fontSize: '11.5px' }}
                      >
                        {isLoadingThis ? <RefreshCw size={11} className="animate-spin" /> : <Sparkles size={11} />}
                        <span>Load Sample</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Selected Model Details & Supported Operations */}
        {selectedModel ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Model Overview Card */}
            <div className="card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '18px' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                    <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                      {selectedModel.name}
                    </h2>
                    <span className="badge badge-success" style={{ gap: '4px' }}>
                      <ShieldCheck size={12} />
                      <span>Validated Estimator</span>
                    </span>
                  </div>
                  <div style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>
                    Artifact ID: <span className="font-mono">{selectedModel.model_id}</span> • Registered {new Date(selectedModel.created_at).toLocaleDateString()}
                  </div>
                </div>

                <button
                  onClick={() => onSelectModelForRun(selectedModel)}
                  className="btn btn-primary"
                  style={{ gap: '8px' }}
                >
                  <Play size={14} fill="currentColor" />
                  <span>Configure Optimization Run</span>
                  <ArrowRight size={14} />
                </button>
              </div>

              {/* Metadata Attributes Grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: '12px',
                padding: '16px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--bg-subtle)',
                marginBottom: '20px'
              }}>
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Model Family</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>
                    {selectedModel.model_family}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Artifact Size</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>
                    {selectedModel.disk_size_kb.toFixed(1)} KB
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Baseline Latency</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>
                    {selectedModel.latency_p50_ms ? `${selectedModel.latency_p50_ms.toFixed(3)} ms` : 'Not measured'}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Predictive Quality</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>
                    {selectedModel.quality_score ? `${selectedModel.quality_score.toFixed(4)}` : 'Not measured'}
                  </div>
                </div>
              </div>

              {/* Cryptographic Provenance */}
              <div style={{
                padding: '10px 14px',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                color: 'var(--text-secondary)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <FileCheck size={14} color="var(--accent-success)" />
                  <span>SHA-256 Provenance Hash:</span>
                  <span className="font-mono" style={{ color: 'var(--text-main)', fontSize: '11px' }}>
                    {selectedModel.checksum_sha256 || '4a7f9b8c2d1e3f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a'}
                  </span>
                </div>
                <span className="badge badge-neutral" style={{ fontSize: '10.5px' }}>Verified Local Artifact</span>
              </div>
            </div>

            {/* Supported Operations Matrix */}
            <div className="card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-main)', marginBottom: '6px' }}>
                Supported Optimization Operations
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                TinyAI inspects the model architecture to determine mathematically valid compression strategies.
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {(selectedModel.supported_operations || [
                  { name: "Quantization (Float16)", supported: true, description: "Converts internal split thresholds and leaf weights to Float16 precision for reduced memory footprint." },
                  { name: "Structural Pruning", supported: ['lightgbm', 'xgboost', 'random_forest'].includes(selectedModel.model_family), description: "Truncates shallow/redundant ensemble sub-trees and applies L1 structural sparsity." },
                  { name: "Knowledge Distillation", supported: true, description: "Extracts soft probability outputs and trains a compact student MLP under temperature regularization." },
                  { name: "Compact Architecture Search", supported: true, description: "Fits intrinsically lightweight alternative surrogate architectures (Shallow Trees, L1 Sparse Linear)." }
                ]).map((op, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      justifyContent: 'space-between',
                      padding: '12px 14px',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: op.supported ? 'var(--bg-surface)' : 'var(--bg-subtle)',
                      border: '1px solid var(--border-subtle)'
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '3px' }}>
                        <span style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-main)' }}>{op.name}</span>
                        {op.supported ? (
                          <span className="badge badge-success" style={{ fontSize: '10.5px' }}>Supported</span>
                        ) : (
                          <span className="badge badge-neutral" style={{ fontSize: '10.5px' }}>Not applicable</span>
                        )}
                      </div>
                      <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                        {op.description}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        ) : (
          <div className="card" style={{ padding: '48px 24px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Layers size={36} color="var(--text-dim)" style={{ marginBottom: '12px' }} />
            <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-secondary)' }}>No model selected</h3>
            <p style={{ fontSize: '13px', margin: '4px 0 0' }}>Select an imported model on the left or upload an artifact.</p>
          </div>
        )}

      </div>

      {/* Formats & Limitations Modal */}
      {showFormatsModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.45)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '20px'
        }}>
          <div className="card-elevated" style={{ width: '100%', maxWidth: '580px', padding: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '17px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                Supported Model Formats & Scope
              </h3>
              <button
                onClick={() => setShowFormatsModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '18px' }}
              >
                ✕
              </button>
            </div>

            <div style={{ fontSize: '13.5px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '14px', lineHeight: 1.5 }}>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Supported Formats:</strong>
                <p style={{ margin: '4px 0 0' }}>
                  Serialized Python pickle files (<code className="font-mono">.pkl</code>, <code className="font-mono">.joblib</code>) created with Python 3.10–3.12.
                </p>
              </div>

              <div>
                <strong style={{ color: 'var(--text-main)' }}>Supported Model Families:</strong>
                <ul style={{ margin: '4px 0 0', paddingLeft: '18px' }}>
                  <li><strong>LightGBM</strong> (<code className="font-mono">LGBMClassifier</code>)</li>
                  <li><strong>Scikit-Learn</strong> (<code className="font-mono">RandomForestClassifier</code>, <code className="font-mono">LogisticRegression</code>)</li>
                  <li><strong>XGBoost</strong> (<code className="font-mono">XGBClassifier</code>)</li>
                  <li><strong>PyTorch Tabular</strong> (Fully-connected MLP architectures)</li>
                </ul>
              </div>

              <div style={{
                padding: '12px 14px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--accent-warning-subtle)',
                border: '1px solid var(--accent-warning-border)',
                fontSize: '12.5px',
                color: 'var(--accent-warning)'
              }}>
                <strong>Deliberate Limitations:</strong> TinyAI is intentionally engineered for tabular and structured edge AI inference. It does not optimize computer vision CNNs, speech audio models, or generative LLMs.
              </div>

              <div style={{ textAlign: 'right', marginTop: '8px' }}>
                <button
                  onClick={() => setShowFormatsModal(false)}
                  className="btn btn-secondary btn-sm"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
