import React from 'react';
import {
  Grid,
  FolderGit2,
  Database,
  Layers,
  Sparkles,
  Award,
  Radio,
  Activity,
  Terminal,
  BookOpen,
  Zap
} from 'lucide-react';

export type NavTab =
  | 'experiment'
  | 'tools_hub'
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
    { id: 'experiment', label: 'Experiment Studio', icon: Zap, badge: 'Core Flow' },
    { id: 'optimization_lab', label: 'Pareto Explorer', icon: Sparkles, badge: 'Trade-offs' },
    { id: 'datasets', label: 'Datasets & Audit', icon: Database },
    { id: 'baselines', label: 'Baseline Training', icon: Layers },
    { id: 'model_registry', label: 'Model Registry', icon: Award },
    { id: 'deployments', label: 'Live Sandbox', icon: Radio },
    { id: 'monitoring', label: 'Drift & Observability', icon: Activity },
    { id: 'tools_hub', label: 'AI Tools Directory', icon: Grid },
    { id: 'projects', label: 'Workspaces', icon: FolderGit2 },
    { id: 'api_sdk', label: 'Python SDK & CLI', icon: Terminal },
    { id: 'documentation', label: 'Docs & Hardware', icon: BookOpen },
  ];

  return (
    <aside style={{
      width: '240px',
      backgroundColor: '#ffffff',
      borderRight: '1px solid #e2e8f0',
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0,
      userSelect: 'none',
      height: 'calc(100vh - 64px)'
    }}>
      {/* Navigation List */}
      <nav style={{ padding: '16px 12px', flex: 1, overflowY: 'auto' }}>
        <div style={{
          fontSize: '11px',
          fontWeight: 700,
          textTransform: 'uppercase',
          color: '#94a3b8',
          padding: '0 12px 10px',
          letterSpacing: '0.06em'
        }}>
          Navigation
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
                    justifyContent: 'space-between',
                    padding: '9px 12px',
                    borderRadius: '8px',
                    border: 'none',
                    backgroundColor: isActive ? '#fef2f2' : 'transparent',
                    color: isActive ? '#e5322d' : '#475569',
                    fontSize: '13.5px',
                    fontWeight: isActive ? 700 : 500,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = '#f8fafc';
                      e.currentTarget.style.color = '#0f172a';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.color = '#475569';
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Icon size={17} color={isActive ? '#e5322d' : '#64748b'} />
                    <span>{item.label}</span>
                  </div>
                  {item.badge && (
                    <span style={{
                      fontSize: '10px',
                      fontWeight: 700,
                      color: isActive ? '#e5322d' : '#64748b',
                      backgroundColor: isActive ? '#fee2e2' : '#f1f5f9',
                      padding: '2px 6px',
                      borderRadius: '6px'
                    }}>
                      {item.badge}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Edge System Summary Box at bottom */}
      <div style={{
        padding: '14px',
        margin: '12px',
        backgroundColor: '#f8fafc',
        borderRadius: '10px',
        border: '1px solid #e2e8f0',
        fontSize: '11px',
        color: '#64748b'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>
          <Zap size={13} color="#e5322d" />
          <span>Local Engine Active</span>
        </div>
        <div>ARM Cortex-M &amp; RPi4 ready</div>
        <div style={{ marginTop: '6px', color: '#059669', fontWeight: 600 }}>
          ● All 4 compression tiers enabled
        </div>
      </div>
    </aside>
  );
};
