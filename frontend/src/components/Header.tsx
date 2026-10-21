import React from 'react';
import {
  Layers,
  Cpu,
  Moon,
  Sun,
  Activity,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  Play
} from 'lucide-react';
import { HealthStatus } from '../api';

export type WorkspaceTab = 'models' | 'runs' | 'results' | 'releases';

interface HeaderProps {
  currentTab: WorkspaceTab;
  onNavigate: (tab: WorkspaceTab) => void;
  backendOnline: boolean;
  healthStatus: HealthStatus | null;
  activeModelName?: string;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onNavigate,
  backendOnline,
  healthStatus,
  activeModelName,
  theme,
  onToggleTheme
}) => {
  return (
    <header style={{
      height: '56px',
      backgroundColor: 'var(--bg-surface)',
      borderBottom: '1px solid var(--border-subtle)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 20px',
      userSelect: 'none',
      zIndex: 100
    }}>
      {/* Left: Brand + Active Model indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div
          onClick={() => onNavigate('runs')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            cursor: 'pointer'
          }}
        >
          <div style={{
            width: '28px',
            height: '28px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'var(--accent-primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff',
            fontWeight: 800,
            fontSize: '14px'
          }}>
            T
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
              TinyAI
            </span>
            <span style={{
              fontSize: '11px',
              color: 'var(--text-muted)',
              borderLeft: '1px solid var(--border-medium)',
              paddingLeft: '6px'
            }}>
              Edge Model Workspace
            </span>
          </div>
        </div>

        {activeModelName && (
          <div
            onClick={() => onNavigate('models')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '3px 10px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'var(--bg-subtle)',
              border: '1px solid var(--border-subtle)',
              fontSize: '12px',
              color: 'var(--text-secondary)',
              cursor: 'pointer'
            }}
            title="Click to view Model details"
          >
            <Layers size={13} color="var(--accent-primary)" />
            <span style={{ fontWeight: 500 }}>{activeModelName}</span>
          </div>
        )}
      </div>

      {/* Right: Actions, Telemetry, and Theme */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* Run Quick Action */}
        <button
          onClick={() => onNavigate('runs')}
          className="btn btn-primary btn-sm"
          style={{ gap: '6px' }}
        >
          <Play size={13} fill="currentColor" />
          <span>New Experiment</span>
        </button>

        {/* Backend Telemetry Chip */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 10px',
          borderRadius: 'var(--radius-sm)',
          backgroundColor: backendOnline ? 'var(--accent-success-subtle)' : 'var(--accent-danger-subtle)',
          border: `1px solid ${backendOnline ? 'var(--accent-success-border)' : 'var(--accent-danger-border)'}`,
          fontSize: '12px',
          color: backendOnline ? 'var(--accent-success)' : 'var(--accent-danger)'
        }}>
          {backendOnline ? (
            <>
              <CheckCircle2 size={13} />
              <span>Engine Ready</span>
              {healthStatus && (
                <span style={{ color: 'var(--text-muted)', fontSize: '11px', marginLeft: '2px' }}>
                  ({healthStatus.cpu_usage_pct}% CPU)
                </span>
              )}
            </>
          ) : (
            <>
              <AlertCircle size={13} />
              <span>Offline</span>
            </>
          )}
        </div>

        {/* Light / Dark Theme Toggle */}
        <button
          onClick={onToggleTheme}
          className="btn btn-ghost btn-sm"
          style={{ padding: '6px 8px', borderRadius: 'var(--radius-sm)' }}
          title={theme === 'dark' ? 'Switch to Light theme' : 'Switch to Dark theme'}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>
    </header>
  );
};
