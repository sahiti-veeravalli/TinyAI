import React from 'react';
import {
  Layers,
  PlayCircle,
  BarChart3,
  PackageCheck,
  Cpu,
  Info,
  FolderGit2
} from 'lucide-react';
import { WorkspaceTab } from './Header';

export type NavTab =
  | WorkspaceTab
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
  currentTab: WorkspaceTab;
  onSelectTab: (tab: WorkspaceTab) => void;
  modelsCount?: number;
  runsCount?: number;
  resultsAvailable?: boolean;
  releasesCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  modelsCount = 0,
  runsCount = 0,
  resultsAvailable = false,
  releasesCount = 0
}) => {
  const navItems = [
    {
      id: 'models' as WorkspaceTab,
      label: 'Models',
      description: 'Imported models & pre-flight audit',
      icon: Layers,
      count: modelsCount > 0 ? modelsCount : undefined
    },
    {
      id: 'runs' as WorkspaceTab,
      label: 'Runs',
      description: 'Objectives & optimization execution',
      icon: PlayCircle,
      count: runsCount > 0 ? runsCount : undefined
    },
    {
      id: 'results' as WorkspaceTab,
      label: 'Results',
      description: 'Candidate comparison & Pareto frontier',
      icon: BarChart3,
      badge: resultsAvailable ? 'Ready' : undefined
    },
    {
      id: 'releases' as WorkspaceTab,
      label: 'Releases',
      description: 'Verified bundles & live edge test',
      icon: PackageCheck,
      count: releasesCount > 0 ? releasesCount : undefined
    }
  ];

  return (
    <aside style={{
      width: '240px',
      backgroundColor: 'var(--bg-surface)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0,
      userSelect: 'none',
      height: 'calc(100vh - 56px)'
    }}>
      {/* Workspace Section Header */}
      <div style={{ padding: '16px 14px 8px' }}>
        <div style={{
          fontSize: '11px',
          fontWeight: 700,
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
          letterSpacing: '0.05em',
          padding: '0 8px 8px'
        }}>
          Workspace
        </div>

        <nav>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <li key={item.id}>
                  <button
                    onClick={() => onSelectTab(item.id)}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 10px',
                      borderRadius: 'var(--radius-md)',
                      border: '1px solid',
                      borderColor: isActive ? 'var(--border-medium)' : 'transparent',
                      backgroundColor: isActive ? 'var(--bg-subtle)' : 'transparent',
                      color: isActive ? 'var(--text-main)' : 'var(--text-secondary)',
                      fontSize: '13.5px',
                      fontWeight: isActive ? 600 : 500,
                      cursor: 'pointer',
                      transition: 'all 0.12s ease',
                      textAlign: 'left'
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'var(--bg-hover)';
                        e.currentTarget.style.color = 'var(--text-main)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                        e.currentTarget.style.color = 'var(--text-secondary)';
                      }
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '9px' }}>
                      <Icon
                        size={16}
                        color={isActive ? 'var(--accent-primary)' : 'var(--text-muted)'}
                      />
                      <span>{item.label}</span>
                    </div>

                    {item.count !== undefined && (
                      <span style={{
                        fontSize: '11px',
                        padding: '1px 6px',
                        borderRadius: '9999px',
                        backgroundColor: isActive ? 'var(--accent-primary-subtle)' : 'var(--bg-muted)',
                        color: isActive ? 'var(--accent-primary)' : 'var(--text-muted)',
                        fontWeight: 600
                      }}>
                        {item.count}
                      </span>
                    )}

                    {item.badge && (
                      <span className="badge badge-orange" style={{ fontSize: '10px', padding: '1px 6px' }}>
                        {item.badge}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>
      </div>

      {/* Bottom info section */}
      <div style={{
        marginTop: 'auto',
        padding: '14px',
        borderTop: '1px solid var(--border-subtle)',
        backgroundColor: 'var(--bg-surface)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px 10px',
          borderRadius: 'var(--radius-sm)',
          backgroundColor: 'var(--bg-subtle)',
          fontSize: '11.5px',
          color: 'var(--text-muted)'
        }}>
          <Cpu size={14} color="var(--text-secondary)" />
          <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.25 }}>
            <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>Host Hardware Engine</span>
            <span>Local microsecond timers</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
