export interface DatasetMetadata {
  dataset_id: string;
  name: string;
  fingerprint: string;
  format: string;
  row_count: number;
  column_count: number;
  memory_bytes: number;
  numeric_columns: string[];
  categorical_columns: string[];
  all_columns: string[];
  preview_rows: Record<string, any>[];
  summary_stats: Record<string, {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
    null_count: number;
  }>;
}

export interface ValidationIssue {
  severity: "ERROR" | "WARNING" | "INFO";
  code: string;
  column?: string;
  message: string;
  recommendation: string;
}

export interface ValidationReport {
  is_valid: boolean;
  task_type: string;
  target_column: string;
  row_count: number;
  column_count: number;
  missing_value_count: number;
  duplicate_rows_count: number;
  class_distribution?: Record<string, number>;
  imbalance_ratio?: number;
  issues: ValidationIssue[];
  feature_columns: string[];
}

export interface CandidateEvaluation {
  candidate_id: string;
  name: string;
  strategy: string;
  model_family: string;
  artifact_path?: string;
  artifact_checksum_sha256?: string;
  quality_metric_name: string;
  quality_score: number;
  latency_p50_ms: number;
  disk_size_kb: number;
  memory_rss_mb: number;
  compression_ratio: number;
  is_feasible: boolean;
  constraint_violations: string[];
  is_pareto_optimal: boolean;
  recommendation_score: number;
  recommendation_rationale?: string;
  benchmark_details: Record<string, any>;
}

export interface ExperimentManifest {
  experiment_id: string;
  project_id: string;
  created_at: string;
  dataset_fingerprint: string;
  target_column: string;
  task_type: string;
  baseline_model_family: string;
  random_seed: number;
  constraints: {
    max_disk_size_kb?: number;
    max_latency_ms?: number;
    min_quality_score?: number;
    max_memory_mb?: number;
  };
  baseline_benchmark: Record<string, any>;
  candidates: CandidateEvaluation[];
  pareto_frontier: CandidateEvaluation[];
  recommended_candidate?: CandidateEvaluation;
  priority: string;
  hardware_environment: Record<string, any>;
  objective_prompt?: string;
}

export interface ModelEntry {
  model_id: string;
  name: string;
  version: string;
  experiment_id: string;
  project_id: string;
  dataset_id: string;
  strategy: string;
  model_family: string;
  framework?: string;
  quality_metric: string;
  quality_score: number;
  latency_p50_ms: number;
  disk_size_kb: number;
  compression_ratio: number;
  artifact_path: string;
  checksum_sha256?: string;
  feature_count?: number;
  is_valid?: boolean;
  validation_status?: string;
  supported_operations?: Array<{
    name: string;
    supported: boolean;
    description: string;
  }>;
  created_at: string;
  status: string;
  bundle_dir?: string;
}

export interface ParsedObjective {
  raw_prompt: string;
  target_hardware: string;
  target_hardware_name: string;
  max_disk_size_kb?: number;
  max_latency_ms?: number;
  min_quality_score?: number;
  quality_metric_name: string;
  max_memory_mb?: number;
  priority: string;
  enabled_strategies: string[];
  detected_entities: Array<{
    entity: string;
    value: string;
    extracted_raw?: string;
  }>;
  interpretation_summary: string;
  validation_warnings: string[];
  is_contradictory: boolean;
  parser_type: string;
  confidence_score: number;
}

export interface SampleModelCatalogItem {
  sample_id: string;
  name: string;
  model_family: string;
  framework: string;
  dataset_name: string;
  baseline_quality: number;
  quality_metric: string;
  baseline_latency_ms: number;
  baseline_size_kb: number;
  feature_count: number;
  target_runtime: string;
  description: string;
  recommended_budget: {
    max_disk_size_kb?: number;
    max_latency_ms?: number;
    min_quality_score?: number;
    priority?: string;
  };
}

export interface ExportBundleResponse {
  model_id: string;
  bundle_directory: string;
  generated_files: Record<string, string>;
  file_checksums?: Record<string, string>;
  reproduction_instructions?: string[];
  message: string;
}

export interface FeatureDriftReport {
  feature_name: string;
  ks_statistic: number;
  p_value: number;
  psi_score: number;
  drift_detected: boolean;
  drift_severity: "NONE" | "LOW" | "MODERATE" | "CRITICAL";
  reference_mean: number;
  current_mean: number;
}

export interface DatasetDriftAnalysis {
  overall_drift_detected: boolean;
  drifted_features_count: number;
  total_features_count: number;
  feature_reports: FeatureDriftReport[];
  recommendation: string;
}

export interface HardwarePreset {
  preset_id: string;
  name: string;
  target_cpu: string;
  recommended_max_size_kb: number;
  recommended_max_latency_ms: number;
  recommended_max_memory_mb: number;
  supported_runtimes: string[];
}

export interface ToolDefinition {
  id: string;
  name: string;
  shortDesc: string;
  badge: string;
  category: 'compression' | 'profiling' | 'deployment' | 'audit';
  icon: string;
  color: string;
  bgLight: string;
  popular?: boolean;
}

export interface FeatureAttributionItem {
  feature: string;
  baseline_importance: number;
  compressed_importance: number;
  delta_pct: number;
}

export interface ModelExplainability {
  model_id: string;
  model_name: string;
  fidelity_score_pct: number;
  rank_correlation: number;
  features_analyzed: number;
  feature_attributions: FeatureAttributionItem[];
  audit_verdict: string;
}
