import React, { useState } from 'react';
import {
  Zap,
  Minimize2,
  Sliders,
  Scissors,
  Activity,
  PlayCircle,
  TrendingDown,
  BookOpen,
  Terminal,
  Grid,
  Database,
  Cpu,
  HardDrive,
  Info,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { NavTab } from './Sidebar';
import { HealthStatus } from '../api';

interface HeaderProps {
  currentTab: string;
  onNavigate: (tab: NavTab) => void;
  onSelectTool: (toolId: string) => void;
  backendOnline: boolean;
  healthStatus?: HealthStatus | null;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onNavigate,
  onSelectTool,
  backendOnline,
  healthStatus
}) => {
  const [showHealthModal, setShowHealthModal] = useState<boolean>(false);

  return (
    <header style={{
      height: '64px',
      backgroundColor: '#ffffff',
      borderBottom: '1px solid #e2e8f0',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)'
    }}>
      {/* Left: Brand Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <div
          onClick={() => onNavigate('experiment')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            cursor: 'pointer',
            userSelect: 'none'
          }}
        >
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '9px',
            backgroundColor: '#e5322d',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 8px rgba(229, 50, 45, 0.3)'
          }}>
            <Zap size={22} color="#ffffff" />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '3px' }}>
            <span style={{ fontSize: '20px', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.03em' }}>
              Tiny<span style={{ color: '#e5322d' }}>AI</span>
            </span>
            <span style={{ fontSize: '10px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase' }}>
              Edge Studio
            </span>
          </div>
        </div>

        {/* Primary Navigation Links */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <button
            onClick={() => onNavigate('experiment')}
            style={{
              padding: '8px 14px',
              borderRadius: '8px',
              border: 'none',
              backgroundColor: currentTab === 'experiment' ? '#fef2f2' : 'transparent',
              color: currentTab === 'experiment' ? '#e5322d' : '#334155',
              fontSize: '13.5px',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Zap size={15} color={currentTab === 'experiment' ? '#e5322d' : '#64748b'} />
            <span>Experiment Studio</span>
          </button>

          <button
            onClick={() => onNavigate('optimization_lab')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: currentTab === 'optimization_lab' ? '#4f46e5' : '#334155',
              backgroundColor: currentTab === 'optimization_lab' ? '#eef2ff' : 'transparent',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <TrendingDown size={14} color="#4f46e5" />
            <span>Pareto Explorer</span>
          </button>

          <button
            onClick={() => onNavigate('datasets')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: currentTab === 'datasets' ? '#0891b2' : '#334155',
              backgroundColor: currentTab === 'datasets' ? '#ecfeff' : 'transparent',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <Database size={14} color="#0891b2" />
            <span>Datasets</span>
          </button>

          <button
            onClick={() => onNavigate('deployments')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: currentTab === 'deployments' ? '#ea580c' : '#334155',
              backgroundColor: currentTab === 'deployments' ? '#fff7ed' : 'transparent',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <PlayCircle size={14} color="#ea580c" />
            <span>Live Sandbox</span>
          </button>

          <button
            onClick={() => onNavigate('tools_hub')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: currentTab === 'tools_hub' ? '#7c3aed' : '#334155',
              backgroundColor: currentTab === 'tools_hub' ? '#f5f3ff' : 'transparent',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <Grid size={14} color="#7c3aed" />
            <span>All AI Tools</span>
          </button>
        </nav>
      </div>

      {/* Right side: Real Backend Health & Shortcuts */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        {/* Real Backend Status Check Pill */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setShowHealthModal(!showHealthModal)}
            title="Click to view real host diagnostics"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '7px',
              padding: '5px 12px',
              borderRadius: '16px',
              backgroundColor: backendOnline ? '#ecfdf5' : '#fef2f2',
              border: `1px solid ${backendOnline ? '#a7f3d0' : '#fecaca'}`,
              fontSize: '12px',
              fontWeight: 600,
              color: backendOnline ? '#059669' : '#dc2626',
              cursor: 'pointer'
            }}
          >
            <span style={{
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              backgroundColor: backendOnline ? '#059669' : '#dc2626'
            }} />
            <span>
              {backendOnline
                ? `Engine Active${healthStatus?.ping_ms ? ` • ${healthStatus.ping_ms}ms` : ''}`
                : 'Backend Offline (Port 8000)'}
            </span>
            <Info size={12} color={backendOnline ? '#059669' : '#dc2626'} />
          </button>

          {/* Real Health Diagnostics Popover */}
          {showHealthModal && (
            <div style={{
              position: 'absolute',
              top: '38px',
              right: 0,
              width: '320px',
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              border: '1px solid #e2e8f0',
              boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
              padding: '16px',
              zIndex: 100
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '13px', fontWeight: 800, color: '#0f172a' }}>
                  Live ML Engine Diagnostics
                </span>
                <button
                  onClick={() => setShowHealthModal(false)}
                  style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: '13px' }}
                >
                  ✕
                </button>
              </div>

              {healthStatus ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ color: '#64748b' }}>Platform:</span>
                    <strong style={{ color: '#0f172a' }}>{healthStatus.platform}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ color: '#64748b' }}>Python Version:</span>
                    <strong style={{ color: '#0f172a' }}>{healthStatus.python_version}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ color: '#64748b' }}>Engine Uptime:</span>
                    <strong style={{ color: '#0f172a' }}>{healthStatus.uptime_seconds}s</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ color: '#64748b' }}>Host CPU Usage:</span>
                    <strong style={{ color: '#2563eb' }}>{healthStatus.cpu_usage_pct}%</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ color: '#64748b' }}>Process RSS Memory:</span>
                    <strong style={{ color: '#059669' }}>{healthStatus.memory_rss_mb} MB</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ color: '#64748b' }}>Inferences Served:</span>
                    <strong style={{ color: '#7c3aed' }}>{healthStatus.inferences_served}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                    <span style={{ color: '#64748b' }}>Active Datasets / Models:</span>
                    <strong style={{ color: '#0f172a' }}>{healthStatus.active_datasets} / {healthStatus.registered_models}</strong>
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: '12px', color: '#dc2626' }}>
                  Cannot connect to FastAPI backend at <code>http://127.0.0.1:8000</code>. Please check terminal task logs.
                </div>
              )}
            </div>
          )}
        </div>

        {/* API SDK shortcut */}
        <button
          onClick={() => onNavigate('api_sdk')}
          style={{
            background: 'none',
            border: 'none',
            color: '#64748b',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '5px'
          }}
        >
          <Terminal size={14} />
          <span>API / SDK</span>
        </button>

        {/* Docs shortcut */}
        <button
          onClick={() => onNavigate('documentation')}
          style={{
            background: 'none',
            border: 'none',
            color: '#64748b',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '5px'
          }}
        >
          <BookOpen size={14} />
          <span>Docs</span>
        </button>

        {/* Action Button: Launch Experiment */}
        <button
          onClick={() => onNavigate('experiment')}
          className="btn btn-primary btn-sm"
          style={{ backgroundColor: '#e5322d' }}
        >
          <span>⚡ Launch Pipeline</span>
        </button>
      </div>
    </header>
  );
};
