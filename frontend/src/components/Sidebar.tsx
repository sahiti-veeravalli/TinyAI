import React from 'react';
import {
  LayoutDashboard,
  FolderGit2,
  Database,
  Layers,
  Sparkles,
  Award,
  Radio,
  Activity,
  Terminal,
  BookOpen,
  Cpu,
  Zap
} from 'lucide-react';

export type NavTab =
  | 'overview'
  | 'projects'
  | 'datasets'
  | 'baselines'
  | 'optimization_lab'
  | 'model_registry'
  | 'deployments'
  | 'monitoring'
  | 'api_sdk'
  | 'documentation';

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  backendOnline: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentTab, onSelectTab, backendOnline }) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'projects', label: 'Projects', icon: FolderGit2 },
    { id: 'datasets', label: 'Datasets', icon: Database },
    { id: 'baselines', label: 'Baseline Models', icon: Layers },
    { id: 'optimization_lab', label: 'Optimization Lab', icon: Sparkles, highlight: true },
    { id: 'model_registry', label: 'Model Registry', icon: Award },
    { id: 'deployments', label: 'Deployments & Sandbox', icon: Radio },
    { id: 'monitoring', label: 'Drift & Observability', icon: Activity },
    { id: 'api_sdk', label: 'Python SDK & CLI', icon: Terminal },
    { id: 'documentation', label: 'Docs & Hardware Presets', icon: BookOpen },
  ];

  return (
    <aside style={{
      width: '260px',
      height: '100vh',
      backgroundColor: '#0a0f1d',
      borderRight: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0,
      userSelect: 'none'
    }}>
      {/* Brand Header */}
      <div style={{
        padding: '20px 20px 16px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{
          width: '36px',
          height: '36px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 16px rgba(99, 102, 241, 0.4)'
        }}>
          <Cpu size={20} color="#fff" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontWeight: 800, fontSize: '17px', letterSpacing: '-0.02em', color: '#fff' }}>
              Tiny<span style={{ color: '#818cf8' }}>AI</span>
            </span>
            <span className="badge badge-indigo" style={{ fontSize: '9px', padding: '1px 5px' }}>v0.1</span>
          </div>
          <p style={{ fontSize: '11px', color: '#64748b' }}>Edge Intelligence Platform</p>
        </div>
      </div>

      {/* Navigation List */}
      <nav style={{ padding: '14px 10px', flex: 1, overflowY: 'auto' }}>
        <div style={{ fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', color: '#475569', padding: '0 10px 8px', letterSpacing: '0.08em' }}>
          Platform Modules
        </div>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <li key={item.id}>
                <button
                  onClick={() => onSelectTab(item.id as NavTab)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '9px 12px',
                    borderRadius: '8px',
                    fontSize: '13px',
                    fontWeight: isActive ? 600 : 500,
                    color: isActive ? '#fff' : item.highlight ? '#a5b4fc' : '#94a3b8',
                    backgroundColor: isActive
                      ? 'rgba(99, 102, 241, 0.2)'
                      : item.highlight
                      ? 'rgba(99, 102, 241, 0.06)'
                      : 'transparent',
                    border: isActive
                      ? '1px solid rgba(99, 102, 241, 0.4)'
                      : item.highlight
                      ? '1px solid rgba(99, 102, 241, 0.15)'
                      : '1px solid transparent',
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = item.highlight ? 'rgba(99, 102, 241, 0.06)' : 'transparent';
                  }}
                >
                  <Icon size={17} color={isActive ? '#818cf8' : item.highlight ? '#818cf8' : '#64748b'} />
                  <span style={{ flex: 1 }}>{item.label}</span>
                  {item.highlight && !isActive && (
                    <span className="badge badge-indigo" style={{ fontSize: '9px', padding: '1px 4px' }}>
                      Core
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Backend Engine Status footer */}
      <div style={{
        padding: '14px 16px',
        borderTop: '1px solid rgba(255, 255, 255, 0.06)',
        backgroundColor: 'rgba(0, 0, 0, 0.2)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
          <span style={{ fontSize: '11px', color: '#64748b', fontWeight: 500 }}>Control Plane API</span>
          <span className={`badge ${backendOnline ? 'badge-green' : 'badge-amber'}`}>
            <span className={backendOnline ? 'pulse-dot' : ''} style={{ width: '6px', height: '6px', marginRight: '3px' }} />
            {backendOnline ? 'Connected' : 'Demo Mode'}
          </span>
        </div>
        <div style={{ fontSize: '11px', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Zap size={12} color="#06b6d4" />
          <span>Local Engine: 127.0.0.1:8000</span>
        </div>
      </div>
    </aside>
  );
};
