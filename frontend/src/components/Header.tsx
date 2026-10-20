import React from 'react';
import { Search, Bell, ShieldCheck, ChevronDown, Cpu, Sparkles } from 'lucide-react';

interface HeaderProps {
  activeProjectName?: string;
  onOpenDocs: () => void;
  onLaunchQuickExp?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeProjectName = 'Point-of-Sale Fraud Sentinel',
  onOpenDocs,
  onLaunchQuickExp
}) => {
  return (
    <header style={{
      height: '60px',
      backgroundColor: '#0a0f1d',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      flexShrink: 0,
      zIndex: 10
    }}>
      {/* Workspace Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 12px',
          backgroundColor: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '8px',
          fontSize: '13px',
          fontWeight: 600,
          color: '#e2e8f0',
          cursor: 'pointer'
        }}>
          <span style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Project:
          </span>
          <span>{activeProjectName}</span>
          <ChevronDown size={14} color="#64748b" />
        </div>

        {/* Edge Target Badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '12px',
          color: '#94a3b8',
          padding: '4px 10px',
          borderRadius: '6px',
          backgroundColor: 'rgba(6, 182, 212, 0.08)',
          border: '1px solid rgba(6, 182, 212, 0.2)'
        }}>
          <Cpu size={14} color="#22d3ee" />
          <span>Target: ARM Cortex-M & RPi4</span>
        </div>
      </div>

      {/* Quick Search & Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          position: 'relative',
          display: 'flex',
          alignItems: 'center'
        }}>
          <Search size={15} color="#64748b" style={{ position: 'absolute', left: '10px' }} />
          <input
            type="text"
            placeholder="Search experiments, models, datasets..."
            style={{
              padding: '6px 12px 6px 32px',
              backgroundColor: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '8px',
              fontSize: '12px',
              color: '#e2e8f0',
              outline: 'none',
              width: '260px'
            }}
          />
        </div>

        {onLaunchQuickExp && (
          <button
            onClick={onLaunchQuickExp}
            className="btn btn-primary btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <Sparkles size={14} />
            <span>Launch Experiment</span>
          </button>
        )}

        {/* Documentation shortcut */}
        <button
          onClick={onOpenDocs}
          style={{
            background: 'none',
            border: 'none',
            color: '#94a3b8',
            fontSize: '13px',
            cursor: 'pointer',
            padding: '6px'
          }}
          title="Open Technical Documentation"
        >
          Docs
        </button>

        {/* Notification Bell */}
        <div style={{
          position: 'relative',
          cursor: 'pointer',
          padding: '6px',
          color: '#94a3b8'
        }}>
          <Bell size={17} />
          <span style={{
            position: 'absolute',
            top: '4px',
            right: '4px',
            width: '6px',
            height: '6px',
            backgroundColor: '#6366f1',
            borderRadius: '50%'
          }} />
        </div>

        {/* User Avatar */}
        <div style={{
          width: '32px',
          height: '32px',
          borderRadius: '50%',
          backgroundColor: '#312e81',
          border: '1px solid rgba(99, 102, 241, 0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 700,
          fontSize: '12px',
          color: '#c7d2fe'
        }}>
          SV
        </div>
      </div>
    </header>
  );
};
