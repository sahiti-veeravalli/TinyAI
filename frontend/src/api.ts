import {
  DatasetMetadata,
  ValidationReport,
  ExperimentManifest,
  ModelEntry,
  DatasetDriftAnalysis,
  HardwarePreset
} from './types';

const API_BASE = '/api';

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchOverview(): Promise<any> {
  const res = await fetch(`${API_BASE}/overview`);
  if (!res.ok) throw new Error('Failed to fetch overview data');
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetMetadata[]> {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function loadSampleDataset(sampleId: string): Promise<{ dataset_id: string; metadata: DatasetMetadata; message: string }> {
  const res = await fetch(`${API_BASE}/datasets/load-sample`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sample_id: sampleId })
  });
  if (!res.ok) throw new Error('Failed to load canonical sample dataset');
  return res.json();
}

export async function validateDataset(datasetId: string, targetColumn: string): Promise<ValidationReport> {
  const res = await fetch(`${API_BASE}/datasets/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, target_column: targetColumn })
  });
  if (!res.ok) throw new Error('Failed to validate dataset');
  return res.json();
}

export async function trainBaseline(datasetId: string, targetColumn: string, modelFamily: string): Promise<any> {
  const res = await fetch(`${API_BASE}/baselines/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, target_column: targetColumn, model_family: modelFamily })
  });
  if (!res.ok) throw new Error('Failed to train baseline model');
  return res.json();
}

export async function launchOptimization(params: {
  dataset_id: string;
  target_column: string;
  baseline_family: string;
  max_disk_size_kb?: number;
  max_latency_ms?: number;
  min_quality_score?: number;
  priority: string;
  enabled_strategies: string[];
}): Promise<ExperimentManifest> {
  const res = await fetch(`${API_BASE}/experiments/launch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  if (!res.ok) throw new Error('Failed to launch optimization experiment');
  return res.json();
}

export async function fetchExperiments(): Promise<ExperimentManifest[]> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error('Failed to fetch experiments');
  return res.json();
}

export async function fetchModels(): Promise<ModelEntry[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function exportModelBundle(modelId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/models/${modelId}/export`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to export model deployment bundle');
  return res.json();
}

export async function predictInference(features: Record<string, any>, modelId?: string): Promise<{
  prediction: number;
  probability?: number;
  latency_ms: number;
  target_column: string;
}> {
  const res = await fetch(`${API_BASE}/inference/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features, model_id: modelId })
  });
  if (!res.ok) throw new Error('Inference request failed');
  return res.json();
}

export async function fetchDriftAnalysis(): Promise<DatasetDriftAnalysis> {
  const res = await fetch(`${API_BASE}/monitoring/drift`);
  if (!res.ok) throw new Error('Failed to fetch drift analysis');
  return res.json();
}

export async function fetchMonitoringMetrics(): Promise<any> {
  const res = await fetch(`${API_BASE}/monitoring/metrics`);
  if (!res.ok) throw new Error('Failed to fetch monitoring metrics');
  return res.json();
}

export async function fetchHardwarePresets(): Promise<HardwarePreset[]> {
  const res = await fetch(`${API_BASE}/hardware-presets`);
  if (!res.ok) throw new Error('Failed to fetch hardware presets');
  return res.json();
}
