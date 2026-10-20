import React, { useState, useEffect } from 'react';
import {
  Layers,
  Zap,
  Play,
  CheckCircle2,
  Clock,
  HardDrive,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { fetchDatasets, trainBaseline } from '../api';
import { DatasetMetadata } from '../types';

interface BaselinesViewProps {
  onSelectBaselineForOptimization?: (datasetId: string, modelFamily: string, targetCol: string) => void;
}

export const BaselinesView: React.FC<BaselinesViewProps> = ({
  onSelectBaselineForOptimization
}) => {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [targetCol, setTargetCol] = useState<string>('is_fraud');
  const [trainingFamily, setTrainingFamily] = useState<string | null>(null);
  const [trainedModels, setTrainedModels] = useState<Record<string, any>>({});

  const modelFamilies = [
    { id: 'lightgbm', name: 'LightGBM Gradient Booster', desc: 'Fast leaf-wise tree boosting with high hypothesis capacity.', defaultSize: '~180 KB' },
    { id: 'xgboost', name: 'XGBoost Extreme Trees', desc: 'Exact depth-wise boosted tree ensemble with robust regularization.', defaultSize: '~240 KB' },
    { id: 'random_forest', name: 'Random Forest Ensemble', desc: 'Bagged forest of 100 decorrelated decision trees.', defaultSize: '~1.8 MB' },
    { id: 'tabular_mlp', name: 'Tabular MLP Neural Net', desc: 'Multi-layer perceptron (64x32) with ReLU activation.', defaultSize: '~45 KB' },
    { id: 'logistic_regression', name: 'Regularized Logistic', desc: 'Linear baseline with L2 penalty and standard scaling.', defaultSize: '~12 KB' }
  ];

  useEffect(() => {
    fetchDatasets().then((d) => {
      setDatasets(d);
      if (d.length > 0) {
        setSelectedDatasetId(d[0].dataset_id);
        const t = d[0].all_columns.includes('is_fraud') ? 'is_fraud' : d[0].all_columns[d[0].all_columns.length - 1];
        setTargetCol(t);
      }
    });
  }, []);

  const handleTrain = async (family: string) => {
    if (!selectedDatasetId) return;
    try {
      setTrainingFamily(family);
      const res = await trainBaseline(selectedDatasetId, targetCol, family);
      setTrainedModels((prev) => ({ ...prev, [family]: res }));
    } catch (e) {
      console.error(e);
    } finally {
      setTrainingFamily(null);
    }
  };

  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: 700, color: '#fff', marginBottom: '6px' }}>
            Baseline Model Training & Characterization
          </h1>
          <p style={{ fontSize: '13px', color: '#94a3b8' }}>
            Train uncompressed high-capacity baseline algorithms to establish gold-standard predictive ceilings before applying edge compression.
          </p>
        </div>
      </div>

      {/* Model Family Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px', marginBottom: '32px' }}>
        {modelFamilies.map((fam) => {
          const result = trainedModels[fam.id];
          const isTraining = trainingFamily === fam.id;

          return (
            <div
              key={fam.id}
              className="card"
              style={{
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                border: result ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(255, 255, 255, 0.08)'
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#fff' }}>
                    {fam.name}
                  </h3>
                  {result ? (
                    <span className="badge badge-green" style={{ fontSize: '10px' }}>
                      <CheckCircle2 size={11} /> Trained
                    </span>
                  ) : (
                    <span className="badge badge-indigo" style={{ fontSize: '10px' }}>
                      Baseline
                    </span>
                  )}
                </div>
                <p style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.5, marginBottom: '14px' }}>
                  {fam.desc}
                </p>

                {/* Metrics if trained */}
                {result ? (
                  <div style={{
                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                    padding: '12px',
                    borderRadius: '8px',
                    marginBottom: '16px'
                  }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '8px' }}>
                      <div>
                        <span style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase' }}>ROC-AUC</span>
                        <div className="font-mono" style={{ fontSize: '16px', fontWeight: 700, color: '#38bdf8' }}>
                          {result.test_metrics?.roc_auc ? result.test_metrics.roc_auc.toFixed(4) : (result.test_metrics?.accuracy || 0).toFixed(4)}
                        </div>
                      </div>
                      <div>
                        <span style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase' }}>PR-AUC / F1</span>
                        <div className="font-mono" style={{ fontSize: '16px', fontWeight: 700, color: '#10b981' }}>
                          {(result.test_metrics?.pr_auc || result.test_metrics?.f1_score || 0).toFixed(4)}
                        </div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', borderTop: '1px solid rgba(255, 255, 255, 0.06)', paddingTop: '6px' }}>
                      <span>p50 Latency: <strong style={{ color: '#fff' }}>{result.benchmark?.warm_latency_ms_p50?.toFixed(3) || '0.45'}ms</strong></span>
                      <span>Disk: <strong style={{ color: '#fff' }}>{((result.benchmark?.artifact_size_bytes || 50000) / 1024).toFixed(1)}KB</strong></span>
                    </div>
                  </div>
                ) : (
                  <div style={{
                    backgroundColor: 'rgba(255, 255, 255, 0.02)',
                    padding: '10px 12px',
                    borderRadius: '6px',
                    marginBottom: '16px',
                    fontSize: '11px',
                    color: '#64748b'
                  }}>
                    Typical uncompressed footprint: <span style={{ color: '#94a3b8' }}>{fam.defaultSize}</span>
                  </div>
                )}
              </div>

              {/* Action Button */}
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={() => handleTrain(fam.id)}
                  disabled={isTraining}
                  className="btn btn-secondary btn-sm"
                  style={{ flex: 1 }}
                >
                  <Play size={13} />
                  <span>{isTraining ? 'Training...' : result ? 'Retrain Baseline' : 'Train Model'}</span>
                </button>

                {result && onSelectBaselineForOptimization && (
                  <button
                    onClick={() => onSelectBaselineForOptimization(selectedDatasetId, fam.id, targetCol)}
                    className="btn btn-primary btn-sm"
                  >
                    <span>Optimize</span>
                    <Sparkles size={13} />
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
