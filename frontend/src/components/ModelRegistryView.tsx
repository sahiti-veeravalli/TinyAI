import React, { useState, useEffect } from 'react';
import {
  Award,
  Download,
  CheckCircle2,
  Package,
  Layers,
  FileCode,
  FolderArchive,
  ArrowRight,
  ExternalLink
} from 'lucide-react';
import { ModelEntry } from '../types';
import { fetchModels, exportModelBundle } from '../api';

export const ModelRegistryView: React.FC = () => {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [exportResult, setExportResult] = useState<any | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const list = await fetchModels();
      setModels(list);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (modelId: string) => {
    try {
      setExportingId(modelId);
      const res = await exportModelBundle(modelId);
      setExportResult(res);
      await loadModels();
    } catch (e) {
      console.error(e);
    } finally {
      setExportingId(null);
    }
  };

  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: 800, color: '#fff' }}>
              Model Registry & Lineage
            </h1>
            <span className="badge badge-indigo">Version Controlled Artifacts</span>
          </div>
          <p style={{ fontSize: '13px', color: '#94a3b8' }}>
            Audited store of compressed model artifacts, hardware fingerprints, provenance manifests, and deployment packages.
          </p>
        </div>
      </div>

      {/* Export notification drawer */}
      {exportResult && (
        <div style={{
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid rgba(16, 185, 129, 0.3)',
          borderRadius: '10px',
          padding: '16px 20px',
          marginBottom: '24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#10b981', fontWeight: 700, fontSize: '14px', marginBottom: '4px' }}>
              <CheckCircle2 size={16} />
              <span>Production Edge Bundle Generated!</span>
            </div>
            <p className="font-mono" style={{ fontSize: '12px', color: '#cbd5e1' }}>
              Bundle Directory: {exportResult.bundle_directory}
            </p>
            <div style={{ display: 'flex', gap: '12px', marginTop: '6px', fontSize: '11px', color: '#94a3b8' }}>
              <span>✓ app.py (FastAPI)</span>
              <span>✓ Dockerfile</span>
              <span>✓ requirements.txt</span>
              <span>✓ load_test.py</span>
              <span>✓ model_card.json</span>
            </div>
          </div>
          <button
            onClick={() => setExportResult(null)}
            className="btn btn-secondary btn-sm"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Models Table */}
      <div className="card" style={{ padding: '24px' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
            <thead>
              <tr style={{ backgroundColor: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <th style={{ padding: '12px 14px', color: '#94a3b8' }}>Model Name & Version</th>
                <th style={{ padding: '12px 14px', color: '#94a3b8' }}>Strategy & Family</th>
                <th style={{ padding: '12px 14px', color: '#94a3b8' }}>Quality</th>
                <th style={{ padding: '12px 14px', color: '#94a3b8' }}>p50 Latency</th>
                <th style={{ padding: '12px 14px', color: '#94a3b8' }}>Disk Footprint</th>
                <th style={{ padding: '12px 14px', color: '#94a3b8' }}>Compression</th>
                <th style={{ padding: '12px 14px', color: '#94a3b8' }}>Status</th>
                <th style={{ padding: '12px 14px', color: '#94a3b8', textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const isExporting = exportingId === m.model_id;
                return (
                  <tr key={m.model_id} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                    <td style={{ padding: '12px 14px' }}>
                      <div style={{ fontWeight: 600, color: '#fff' }}>{m.name}</div>
                      <span className="font-mono" style={{ fontSize: '11px', color: '#64748b' }}>
                        {m.model_id} • v{m.version}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <div style={{ color: '#cbd5e1' }}>{m.strategy}</div>
                      <span style={{ fontSize: '11px', color: '#64748b' }}>{m.model_family}</span>
                    </td>
                    <td className="font-mono" style={{ padding: '12px 14px', color: '#38bdf8', fontWeight: 600 }}>
                      {m.quality_score.toFixed(4)}
                    </td>
                    <td className="font-mono" style={{ padding: '12px 14px', color: '#22d3ee' }}>
                      {m.latency_p50_ms.toFixed(3)} ms
                    </td>
                    <td className="font-mono" style={{ padding: '12px 14px', color: '#34d399' }}>
                      {m.disk_size_kb.toFixed(1)} KB
                    </td>
                    <td className="font-mono" style={{ padding: '12px 14px', color: '#a78bfa' }}>
                      {m.compression_ratio.toFixed(1)}x
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span className={`badge ${m.status === 'exported' ? 'badge-green' : 'badge-indigo'}`}>
                        {m.status === 'exported' ? 'Exported' : 'Ready'}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'right' }}>
                      <button
                        onClick={() => handleExport(m.model_id)}
                        disabled={isExporting}
                        className="btn btn-secondary btn-sm"
                        style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                      >
                        <Package size={13} color="#818cf8" />
                        <span>{isExporting ? 'Generating Bundle...' : '1-Click Export Bundle'}</span>
                      </button>
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
