import React, { useState, useEffect } from 'react';
import {
  Database,
  Upload,
  AlertTriangle,
  CheckCircle2,
  Table,
  BarChart2,
  ShieldAlert,
  ArrowRight,
  Info,
  RefreshCw,
  Sparkles
} from 'lucide-react';
import { DatasetMetadata, ValidationReport } from '../types';
import { fetchDatasets, loadSampleDataset, validateDataset } from '../api';

interface DatasetsViewProps {
  onDatasetSelect?: (datasetId: string, targetCol: string) => void;
}

export const DatasetsView: React.FC<DatasetsViewProps> = ({ onDatasetSelect }) => {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingSample, setLoadingSample] = useState<boolean>(false);
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const list = await fetchDatasets();
      setDatasets(list);
      if (list.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(list[0].dataset_id);
        const defaultTarget = list[0].all_columns.includes('is_fraud')
          ? 'is_fraud'
          : list[0].all_columns.includes('maintenance_required')
          ? 'maintenance_required'
          : list[0].all_columns[list[0].all_columns.length - 1];
        setTargetColumn(defaultTarget);
        runValidation(list[0].dataset_id, defaultTarget);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async (sampleId: string) => {
    try {
      setLoadingSample(true);
      const res = await loadSampleDataset(sampleId);
      await loadData();
      setSelectedDatasetId(res.dataset_id);
      const target = sampleId === 'fraud_detection_benchmark' ? 'is_fraud' : 'maintenance_required';
      setTargetColumn(target);
      runValidation(res.dataset_id, target);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingSample(false);
    }
  };

  const runValidation = async (dsId: string, target: string) => {
    try {
      const rep = await validateDataset(dsId, target);
      setValidationReport(rep);
    } catch (e) {
      console.error(e);
    }
  };

  const currentDataset = datasets.find((d) => d.dataset_id === selectedDatasetId);

  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: 700, color: '#fff', marginBottom: '6px' }}>
            Tabular Datasets & Integrity Validation
          </h1>
          <p style={{ fontSize: '13px', color: '#94a3b8' }}>
            Safely ingest untrusted CSV/Parquet files, inspect schema, check class imbalance, and detect leakage channels.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => handleLoadSample('fraud_detection_benchmark')}
            disabled={loadingSample}
            className="btn btn-secondary btn-sm"
          >
            <Sparkles size={14} color="#818cf8" />
            <span>Load Fraud Benchmark (Imbalanced)</span>
          </button>
          <button
            onClick={() => handleLoadSample('edge_iot_predictive_maintenance')}
            disabled={loadingSample}
            className="btn btn-secondary btn-sm"
          >
            <Sparkles size={14} color="#06b6d4" />
            <span>Load IoT Telemetry</span>
          </button>
        </div>
      </div>

      {/* Dataset Selection Tabs */}
      {datasets.length > 0 && (
        <div style={{
          display: 'flex',
          gap: '8px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          marginBottom: '20px',
          paddingBottom: '8px',
          overflowX: 'auto'
        }}>
          {datasets.map((d) => (
            <button
              key={d.dataset_id}
              onClick={() => {
                setSelectedDatasetId(d.dataset_id);
                const target = d.all_columns.includes('is_fraud')
                  ? 'is_fraud'
                  : d.all_columns.includes('maintenance_required')
                  ? 'maintenance_required'
                  : d.all_columns[d.all_columns.length - 1];
                setTargetColumn(target);
                runValidation(d.dataset_id, target);
              }}
              style={{
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '13px',
                fontWeight: selectedDatasetId === d.dataset_id ? 600 : 500,
                color: selectedDatasetId === d.dataset_id ? '#fff' : '#94a3b8',
                backgroundColor: selectedDatasetId === d.dataset_id ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
                border: selectedDatasetId === d.dataset_id ? '1px solid rgba(99, 102, 241, 0.4)' : '1px solid transparent',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                whiteSpace: 'nowrap'
              }}
            >
              <Database size={15} color={selectedDatasetId === d.dataset_id ? '#818cf8' : '#64748b'} />
              <span>{d.name}</span>
              <span className="badge badge-indigo" style={{ fontSize: '10px' }}>
                {d.row_count.toLocaleString()} rows
              </span>
            </button>
          ))}
        </div>
      )}

      {currentDataset && (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginBottom: '24px' }}>
          {/* Main Inspection Card */}
          <div className="card" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#fff', marginBottom: '4px' }}>
                  {currentDataset.name}
                </h2>
                <span className="font-mono" style={{ fontSize: '11px', color: '#64748b' }}>
                  SHA-256: {currentDataset.fingerprint.slice(0, 16)}...{currentDataset.fingerprint.slice(-8)}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span className="badge badge-cyan">{currentDataset.numeric_columns.length} Numeric</span>
                <span className="badge badge-indigo">{currentDataset.categorical_columns.length} Categorical</span>
                <span className="badge badge-green">{(currentDataset.memory_bytes / 1024).toFixed(1)} KB RAM</span>
              </div>
            </div>

            {/* Target Column Selector */}
            <div style={{
              backgroundColor: 'rgba(255, 255, 255, 0.03)',
              padding: '14px 16px',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <div>
                <label style={{ fontSize: '12px', fontWeight: 600, color: '#e2e8f0', display: 'block', marginBottom: '2px' }}>
                  Optimization Target Feature
                </label>
                <p style={{ fontSize: '11px', color: '#94a3b8' }}>
                  Defines classification ground truth label for model training and edge evaluation.
                </p>
              </div>
              <select
                value={targetColumn}
                onChange={(e) => {
                  setTargetColumn(e.target.value);
                  runValidation(currentDataset.dataset_id, e.target.value);
                }}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#0a0f1d',
                  border: '1px solid rgba(99, 102, 241, 0.4)',
                  borderRadius: '6px',
                  color: '#fff',
                  fontSize: '13px',
                  outline: 'none'
                }}
              >
                {currentDataset.all_columns.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            {/* Preview Table */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
                <Table size={15} color="#818cf8" />
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#e2e8f0' }}>
                  Data Stream Preview (First 5 Rows)
                </span>
              </div>
              <div style={{ overflowX: 'auto', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '8px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ backgroundColor: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
                      {currentDataset.all_columns.slice(0, 8).map((col) => (
                        <th key={col} style={{ padding: '8px 12px', color: col === targetColumn ? '#818cf8' : '#94a3b8', fontWeight: 600 }}>
                          {col} {col === targetColumn && '(Target)'}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {currentDataset.preview_rows.slice(0, 5).map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                        {currentDataset.all_columns.slice(0, 8).map((col) => (
                          <td key={col} className="font-mono" style={{ padding: '7px 12px', color: col === targetColumn ? '#a5b4fc' : '#cbd5e1' }}>
                            {String(row[col])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Validation & Data Leakage Panel */}
          <div className="card" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
              <ShieldAlert size={18} color="#f59e0b" />
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#fff' }}>
                Integrity & Leakage Report
              </h3>
            </div>

            {validationReport ? (
              <div>
                <div style={{
                  padding: '12px',
                  borderRadius: '8px',
                  backgroundColor: validationReport.is_valid ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                  border: `1px solid ${validationReport.is_valid ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                  marginBottom: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  <CheckCircle2 size={16} color={validationReport.is_valid ? '#10b981' : '#ef4444'} />
                  <span style={{ fontSize: '13px', fontWeight: 600, color: validationReport.is_valid ? '#34d399' : '#f87171' }}>
                    {validationReport.is_valid ? 'Dataset Valid for Optimization' : 'Blocking Schema Errors Found'}
                  </span>
                </div>

                {/* Class Imbalance Metric */}
                {validationReport.imbalance_ratio && (
                  <div style={{ marginBottom: '14px', padding: '10px 12px', backgroundColor: 'rgba(255, 255, 255, 0.03)', borderRadius: '6px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>
                      <span>Class Imbalance Ratio</span>
                      <span className="font-mono" style={{ color: validationReport.imbalance_ratio > 10 ? '#f59e0b' : '#10b981', fontWeight: 700 }}>
                        {validationReport.imbalance_ratio.toFixed(1)}:1
                      </span>
                    </div>
                    {validationReport.imbalance_ratio > 10 && (
                      <p style={{ fontSize: '11px', color: '#f59e0b', lineHeight: 1.4 }}>
                        Severe imbalance. Accuracy is misleading; PR-AUC and ROC-AUC will be used as primary metrics.
                      </p>
                    )}
                  </div>
                )}

                {/* Issues List */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {validationReport.issues.map((issue, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '10px',
                        borderRadius: '6px',
                        backgroundColor: issue.severity === 'WARNING' ? 'rgba(245, 158, 11, 0.06)' : 'rgba(255, 255, 255, 0.02)',
                        border: `1px solid ${issue.severity === 'WARNING' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(255, 255, 255, 0.06)'}`
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                        <span className={`badge ${issue.severity === 'WARNING' ? 'badge-amber' : 'badge-indigo'}`} style={{ fontSize: '9px' }}>
                          {issue.code}
                        </span>
                        {issue.column && (
                          <span style={{ fontSize: '11px', color: '#cbd5e1', fontWeight: 600 }}>{issue.column}</span>
                        )}
                      </div>
                      <p style={{ fontSize: '11px', color: '#94a3b8', margin: '4px 0' }}>{issue.message}</p>
                      <p style={{ fontSize: '11px', color: '#818cf8', fontWeight: 500 }}>Rec: {issue.recommendation}</p>
                    </div>
                  ))}
                </div>

                {onDatasetSelect && (
                  <button
                    onClick={() => onDatasetSelect(currentDataset.dataset_id, targetColumn)}
                    className="btn btn-primary"
                    style={{ width: '100%', marginTop: '20px' }}
                  >
                    <span>Proceed to Baseline Training</span>
                    <ArrowRight size={15} />
                  </button>
                )}
              </div>
            ) : (
              <p style={{ fontSize: '12px', color: '#64748b' }}>Select a target column to execute integrity report.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
