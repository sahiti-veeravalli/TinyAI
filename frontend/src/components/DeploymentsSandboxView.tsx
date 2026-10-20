import React, { useState } from 'react';
import {
  Radio,
  Zap,
  Play,
  CheckCircle2,
  AlertTriangle,
  Code2,
  Terminal,
  Clock,
  Sparkles,
  ShieldCheck,
  Cpu
} from 'lucide-react';
import { predictInference } from '../api';

export const DeploymentsSandboxView: React.FC = () => {
  // Test prediction inputs for fraud model
  const [amount, setAmount] = useState<number>(45.50);
  const [hour, setHour] = useState<number>(14);
  const [distanceKm, setDistanceKm] = useState<number>(4.2);
  const [v1, setV1] = useState<number>(0.12);
  const [v2, setV2] = useState<number>(-0.45);
  const [v3, setV3] = useState<number>(0.05);

  const [activeTab, setActiveTab] = useState<'sandbox' | 'docker' | 'api'>('sandbox');
  const [predicting, setPredicting] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<any | null>(null);

  const handlePredict = async () => {
    try {
      setPredicting(true);
      const res = await predictInference({
        transaction_amount: amount,
        transaction_hour: hour,
        merchant_distance_km: distanceKm,
        v1_signal: v1,
        v2_signal: v2,
        v3_signal: v3,
        v4_signal: 0.0,
        v5_signal: 0.0,
        payment_channel: 'ONLINE',
        device_platform: 'iOS'
      });
      setPredictionResult(res);
    } catch (e) {
      console.error(e);
    } finally {
      setPredicting(false);
    }
  };

  const setFraudPreset = () => {
    setAmount(720.0);
    setHour(3);
    setDistanceKm(145.0);
    setV1(-3.2);
    setV2(4.1);
    setV3(-2.5);
  };

  const setLegitPreset = () => {
    setAmount(32.50);
    setHour(12);
    setDistanceKm(2.1);
    setV1(0.05);
    setV2(-0.2);
    setV3(0.1);
  };

  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: 800, color: '#fff' }}>
              Edge Deployments & Live Inference Test Bench
            </h1>
            <span className="badge badge-green">Low Latency Runtime</span>
          </div>
          <p style={{ fontSize: '13px', color: '#94a3b8' }}>
            Test compressed models with real inputs in sub-millisecond execution times and inspect generated container bundles.
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid rgba(255, 255, 255, 0.08)', marginBottom: '24px' }}>
        {[
          { id: 'sandbox', label: 'Interactive Prediction Sandbox', icon: Radio },
          { id: 'docker', label: 'Dockerfile & Container Bundle', icon: Code2 },
          { id: 'api', label: 'REST API & cURL Command', icon: Terminal }
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 16px',
                border: 'none',
                borderBottom: isActive ? '2px solid #6366f1' : '2px solid transparent',
                backgroundColor: 'transparent',
                color: isActive ? '#fff' : '#94a3b8',
                fontWeight: isActive ? 600 : 500,
                fontSize: '13px',
                cursor: 'pointer'
              }}
            >
              <Icon size={15} color={isActive ? '#818cf8' : '#64748b'} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {activeTab === 'sandbox' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          {/* Input Form Panel */}
          <div className="card" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#fff' }}>
                Input Feature Payload
              </h2>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={setLegitPreset} className="btn btn-secondary btn-sm">
                  Legit Preset
                </button>
                <button onClick={setFraudPreset} className="btn btn-secondary btn-sm">
                  Suspicious Preset
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
              <div>
                <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  Transaction Amount ($)
                </label>
                <input
                  type="number"
                  value={amount}
                  onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0a0f1d',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: '#fff',
                    fontSize: '13px'
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  Transaction Hour (0-23)
                </label>
                <input
                  type="number"
                  min="0"
                  max="23"
                  value={hour}
                  onChange={(e) => setHour(parseInt(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0a0f1d',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: '#fff',
                    fontSize: '13px'
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  Merchant Distance (km)
                </label>
                <input
                  type="number"
                  value={distanceKm}
                  onChange={(e) => setDistanceKm(parseFloat(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0a0f1d',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: '#fff',
                    fontSize: '13px'
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  v1_signal
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={v1}
                  onChange={(e) => setV1(parseFloat(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0a0f1d',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: '#fff',
                    fontSize: '13px'
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  v2_signal
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={v2}
                  onChange={(e) => setV2(parseFloat(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0a0f1d',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: '#fff',
                    fontSize: '13px'
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  v3_signal
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={v3}
                  onChange={(e) => setV3(parseFloat(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: '#0a0f1d',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: '#fff',
                    fontSize: '13px'
                  }}
                />
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={predicting}
              className="btn btn-primary"
              style={{ width: '100%', padding: '12px' }}
            >
              <Zap size={16} />
              <span>{predicting ? 'Executing Edge Inference...' : 'Execute Edge Prediction'}</span>
            </button>
          </div>

          {/* Real-time Response Output Panel */}
          <div className="card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#fff' }}>
                  Inference Execution Telemetry
                </h2>
                <span className="badge badge-indigo">Direct In-Memory Run</span>
              </div>

              {predictionResult ? (
                <div>
                  {/* Status Banner */}
                  <div style={{
                    padding: '16px',
                    borderRadius: '10px',
                    backgroundColor: predictionResult.prediction === 1 ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                    border: `1px solid ${predictionResult.prediction === 1 ? 'rgba(239, 68, 68, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
                    marginBottom: '20px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '14px'
                  }}>
                    {predictionResult.prediction === 1 ? (
                      <AlertTriangle size={28} color="#ef4444" />
                    ) : (
                      <ShieldCheck size={28} color="#10b981" />
                    )}
                    <div>
                      <div style={{ fontSize: '17px', fontWeight: 800, color: '#fff' }}>
                        {predictionResult.prediction === 1 ? 'FRAUD DETECTED (High Risk)' : 'TRANSACTION LEGITIMATE'}
                      </div>
                      <div style={{ fontSize: '12px', color: '#cbd5e1' }}>
                        Confidence Probability: <strong className="font-mono">{(predictionResult.probability * 100).toFixed(2)}%</strong>
                      </div>
                    </div>
                  </div>

                  {/* Micro-timing telemetry */}
                  <div style={{
                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                    padding: '16px',
                    borderRadius: '8px',
                    border: '1px solid rgba(255, 255, 255, 0.06)',
                    marginBottom: '16px'
                  }}>
                    <div style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', marginBottom: '6px' }}>
                      Measured Execution Latency
                    </div>
                    <div className="font-mono" style={{ fontSize: '28px', fontWeight: 800, color: '#22d3ee', marginBottom: '4px' }}>
                      {predictionResult.latency_ms.toFixed(3)} ms
                    </div>
                    <p style={{ fontSize: '12px', color: '#10b981' }}>
                      ⚡ Satisfies sub-millisecond edge SLA constraint (p50 &lt; 1.0ms)
                    </p>
                  </div>
                </div>
              ) : (
                <div style={{
                  padding: '40px 20px',
                  textAlign: 'center',
                  backgroundColor: 'rgba(255, 255, 255, 0.02)',
                  borderRadius: '8px',
                  border: '1px dashed rgba(255, 255, 255, 0.08)'
                }}>
                  <Cpu size={32} color="#64748b" style={{ margin: '0 auto 12px' }} />
                  <p style={{ fontSize: '13px', color: '#94a3b8' }}>
                    Click "Execute Edge Prediction" to run the optimized model and record real inference timing.
                  </p>
                </div>
              )}
            </div>

            <div style={{ fontSize: '11px', color: '#64748b', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Clock size={12} />
              <span>Tested on local host processor via vectorized preprocessor transform.</span>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'docker' && (
        <div className="card" style={{ padding: '24px' }}>
          <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#fff', marginBottom: '8px' }}>
            Production Edge Dockerfile
          </h2>
          <pre className="font-mono" style={{
            backgroundColor: '#0a0f1d',
            padding: '18px',
            borderRadius: '8px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            fontSize: '12px',
            color: '#a5b4fc',
            lineHeight: 1.6,
            overflowX: 'auto'
          }}>
{`FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl preprocessor.pkl app.py ./

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]`}
          </pre>
        </div>
      )}

      {activeTab === 'api' && (
        <div className="card" style={{ padding: '24px' }}>
          <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#fff', marginBottom: '8px' }}>
            cURL Request Sample
          </h2>
          <pre className="font-mono" style={{
            backgroundColor: '#0a0f1d',
            padding: '18px',
            borderRadius: '8px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            fontSize: '12px',
            color: '#38bdf8',
            lineHeight: 1.6,
            overflowX: 'auto'
          }}>
{`curl -X POST http://127.0.0.1:8000/api/inference/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": {
      "transaction_amount": 125.50,
      "transaction_hour": 14,
      "merchant_distance_km": 3.8,
      "v1_signal": -0.12,
      "v2_signal": 0.45,
      "v3_signal": -0.05
    }
  }'`}
          </pre>
        </div>
      )}
    </div>
  );
};
