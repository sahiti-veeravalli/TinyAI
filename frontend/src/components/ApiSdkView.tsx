import React, { useState } from 'react';
import { Terminal, Code2, Copy, Check, ExternalLink, Cpu } from 'lucide-react';

export const ApiSdkView: React.FC = () => {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const copyCode = (key: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const pythonSnippet = `from tinyai_sdk import TinyAIClient

# Connect to TinyAI Engine
client = TinyAIClient(base_url="http://127.0.0.1:8000")

# 1. Ingest & validate dataset
sample = client.load_sample_dataset("fraud_detection_benchmark")
dataset_id = sample["dataset_id"]

# 2. Launch Pareto optimization under strict edge SLA
experiment = client.launch_optimization(
    dataset_id=dataset_id,
    target_column="is_fraud",
    baseline_family="lightgbm",
    max_disk_size_kb=250.0,
    max_latency_ms=1.0,
    min_quality_score=0.85,
    priority="balanced"
)

# 3. Inspect recommended model & export bundle
recommended = experiment["recommended_candidate"]
print(f"Optimal Model: {recommended['name']} | p50: {recommended['latency_p50_ms']}ms")

bundle = client.export_bundle(model_id=recommended["candidate_id"])
print(f"Docker bundle generated at: {bundle['bundle_directory']}")
`;

  const cliSnippet = `# Check engine status and local hardware environment
tinyai-cli health

# List ingested datasets
tinyai-cli datasets

# Launch headless optimization experiment for CI/CD pipeline
tinyai-cli optimize \\
  --dataset-id ds_fraud_canonical \\
  --target is_fraud \\
  --baseline lightgbm \\
  --max-size-kb 200.0 \\
  --max-lat-ms 1.0 \\
  --priority balanced

# Export production container bundle
tinyai-cli export --model-id mod_canonical_pruned
`;

  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a' }}>
              Developer Platform: Python SDK & CLI
            </h1>
            <span className="badge badge-indigo">Programmatic Automation</span>
          </div>
          <p style={{ fontSize: '13px', color: '#64748b' }}>
            Integrate TinyAI budget-constrained compression directly into MLOps pipelines, GitHub Actions, and embedded runtimes.
          </p>
        </div>
        <a
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-secondary btn-sm"
          style={{ textDecoration: 'none' }}
        >
          <span>OpenAPI / Swagger UI</span>
          <ExternalLink size={13} />
        </a>
      </div>

      {/* Python SDK */}
      <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Code2 size={18} color="#818cf8" />
            <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>
              Python SDK (`tinyai_sdk`)
            </h2>
          </div>
          <button
            onClick={() => copyCode('py', pythonSnippet)}
            className="btn btn-secondary btn-sm"
          >
            {copiedKey === 'py' ? <Check size={13} color="#10b981" /> : <Copy size={13} />}
            <span>{copiedKey === 'py' ? 'Copied!' : 'Copy Code'}</span>
          </button>
        </div>
        <pre className="font-mono" style={{
          backgroundColor: '#ffffff',
          padding: '18px',
          borderRadius: '8px',
          border: '1px solid #e2e8f0',
          fontSize: '12px',
          color: '#475569',
          lineHeight: 1.6,
          overflowX: 'auto'
        }}>
          {pythonSnippet}
        </pre>
      </div>

      {/* CLI Tool */}
      <div className="card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Terminal size={18} color="#06b6d4" />
            <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>
              TinyAI CLI (`tinyai-cli`)
            </h2>
          </div>
          <button
            onClick={() => copyCode('cli', cliSnippet)}
            className="btn btn-secondary btn-sm"
          >
            {copiedKey === 'cli' ? <Check size={13} color="#10b981" /> : <Copy size={13} />}
            <span>{copiedKey === 'cli' ? 'Copied!' : 'Copy Commands'}</span>
          </button>
        </div>
        <pre className="font-mono" style={{
          backgroundColor: '#ffffff',
          padding: '18px',
          borderRadius: '8px',
          border: '1px solid #e2e8f0',
          fontSize: '12px',
          color: '#38bdf8',
          lineHeight: 1.6,
          overflowX: 'auto'
        }}>
          {cliSnippet}
        </pre>
      </div>
    </div>
  );
};
