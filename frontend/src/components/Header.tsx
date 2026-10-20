import React from 'react';
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
  ChevronDown
} from 'lucide-react';
import { NavTab } from './Sidebar';

interface HeaderProps {
  currentTab: string;
  onNavigate: (tab: NavTab) => void;
  onSelectTool: (toolId: string) => void;
  backendOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onNavigate,
  onSelectTool,
  backendOnline
}) => {
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
      {/* Left: Brand Logo — iLovePDF style with red accent */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
        <div
          onClick={() => onNavigate('overview')}
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
            <span style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
              Edge
            </span>
          </div>
        </div>

        {/* Quick Nav Links — iLovePDF style */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <button
            onClick={() => onNavigate('overview')}
            style={{
              padding: '8px 14px',
              borderRadius: '8px',
              border: 'none',
              backgroundColor: currentTab === 'overview' ? '#fef2f2' : 'transparent',
              color: currentTab === 'overview' ? '#e5322d' : '#334155',
              fontSize: '13.5px',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Grid size={15} color={currentTab === 'overview' ? '#e5322d' : '#64748b'} />
            <span>All AI Tools</span>
          </button>

          <button
            onClick={() => onSelectTool('compress')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: '#334155',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <Minimize2 size={14} color="#e5322d" />
            <span>Compress Model</span>
          </button>

          <button
            onClick={() => onSelectTool('quantize')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: '#334155',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <Sliders size={14} color="#059669" />
            <span>Quantize</span>
          </button>

          <button
            onClick={() => onSelectTool('prune')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: '#334155',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <Scissors size={14} color="#7c3aed" />
            <span>Prune</span>
          </button>

          <button
            onClick={() => onSelectTool('benchmark')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: '#334155',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <Activity size={14} color="#2563eb" />
            <span>Benchmark</span>
          </button>

          <button
            onClick={() => onNavigate('deployments')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: currentTab === 'deployments' ? '#ea580c' : '#334155',
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
            onClick={() => onNavigate('optimization_lab')}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'none',
              color: currentTab === 'optimization_lab' ? '#4f46e5' : '#334155',
              fontSize: '13.5px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <TrendingDown size={14} color="#4f46e5" />
            <span>Pareto Lab</span>
          </button>
        </nav>
      </div>

      {/* Right side status & shortcuts */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {/* Backend Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 10px',
          borderRadius: '16px',
          backgroundColor: backendOnline ? '#ecfdf5' : '#fef2f2',
          border: `1px solid ${backendOnline ? '#a7f3d0' : '#fecaca'}`,
          fontSize: '12px',
          fontWeight: 600,
          color: backendOnline ? '#059669' : '#dc2626'
        }}>
          <span style={{
            width: '7px',
            height: '7px',
            borderRadius: '50%',
            backgroundColor: backendOnline ? '#059669' : '#dc2626'
          }} />
          <span>{backendOnline ? 'ML Engine Online' : 'Offline'}</span>
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

        {/* Action Button: Quick Compress */}
        <button
          onClick={() => onSelectTool('compress')}
          className="btn btn-primary btn-sm"
          style={{ backgroundColor: '#e5322d' }}
        >
          <span>⚡ Compress Model</span>
        </button>
      </div>
    </header>
  );
};
