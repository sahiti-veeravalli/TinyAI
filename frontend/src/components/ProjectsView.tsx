import React, { useState } from 'react';
import { FolderGit2, Plus, ArrowRight, Cpu, Clock, CheckCircle2 } from 'lucide-react';
import { NavTab } from './Sidebar';

interface ProjectsViewProps {
  onNavigate: (tab: NavTab) => void;
  projects: any[];
}

export const ProjectsView: React.FC<ProjectsViewProps> = ({ onNavigate, projects }) => {
  return (
    <div className="animate-fade-in" style={{ padding: '28px 36px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a' }}>
              Project Workspaces
            </h1>
            <span className="badge badge-indigo">Multi-Tenant Isolation</span>
          </div>
          <p style={{ fontSize: '13px', color: '#64748b' }}>
            Manage optimization targets, resource quotas, and experiment lineages grouped by deployment domain.
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
        {projects.map((proj) => (
          <div key={proj.project_id} className="card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>
                  {proj.name}
                </h3>
                <span className="badge badge-cyan" style={{ fontSize: '10px' }}>
                  {proj.target_runtime}
                </span>
              </div>
              <p style={{ fontSize: '12px', color: '#64748b', lineHeight: 1.6, marginBottom: '16px' }}>
                {proj.description}
              </p>
              <div style={{
                backgroundColor: '#f8fafc', border: '1px solid #e2e8f0',
                padding: '10px 12px',
                borderRadius: '6px',
                fontSize: '11px',
                display: 'flex',
                justifyContent: 'space-between',
                color: '#475569',
                marginBottom: '16px'
              }}>
                <span>Experiments: <strong style={{ color: '#0f172a' }}>{proj.experiments_count || 1}</strong></span>
                <span>Production Models: <strong style={{ color: '#10b981' }}>{proj.models_count || 2}</strong></span>
              </div>
            </div>

            <button
              onClick={() => onNavigate('optimization_lab')}
              className="btn btn-secondary btn-sm"
              style={{ width: '100%', justifyContent: 'space-between' }}
            >
              <span>Open Optimization Lab</span>
              <ArrowRight size={14} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
