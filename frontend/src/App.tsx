import React, { useState, useEffect } from 'react';
import { Header, WorkspaceTab } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ModelsWorkspaceView } from './components/ModelsWorkspaceView';
import { RunsWorkspaceView } from './components/RunsWorkspaceView';
import { ResultsWorkspaceView } from './components/ResultsWorkspaceView';
import { ReleasesWorkspaceView } from './components/ReleasesWorkspaceView';
import {
  fetchHealthDetails,
  fetchModels,
  fetchExperiments,
  HealthStatus
} from './api';
import { ModelEntry, ExperimentManifest } from './types';

export const App: React.FC = () => {
  // Navigation & Workspace State
  const [currentTab, setCurrentTab] = useState<WorkspaceTab>('runs');
  const [activeModel, setActiveModel] = useState<ModelEntry | null>(null);
  const [currentManifest, setCurrentManifest] = useState<ExperimentManifest | null>(null);
  const [selectedReleaseModelId, setSelectedReleaseModelId] = useState<string | undefined>(undefined);

  // System & Health State
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [modelsCount, setModelsCount] = useState<number>(0);
  const [runsCount, setRunsCount] = useState<number>(0);

  // Theme State
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('tinyai_theme');
    if (saved === 'dark' || saved === 'light') return saved;
    return 'light'; // Default direction is warm-neutral light
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('tinyai_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  useEffect(() => {
    const syncStatusAndWorkspace = async () => {
      const details = await fetchHealthDetails();
      setHealthStatus(details);
      setBackendOnline(!!details);

      if (details) {
        try {
          const [modelsList, experimentsList] = await Promise.all([
            fetchModels(),
            fetchExperiments()
          ]);
          setModelsCount(modelsList.length);
          setRunsCount(experimentsList.length);

          if (!activeModel && modelsList.length > 0) {
            setActiveModel(modelsList[0]);
          }

          if (!currentManifest && experimentsList.length > 0) {
            setCurrentManifest(experimentsList[0]);
          }
        } catch (e) {
          console.error('Error fetching workspace counts:', e);
        }
      }
    };

    syncStatusAndWorkspace();
    const interval = setInterval(syncStatusAndWorkspace, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectModelForRun = (model: ModelEntry) => {
    setActiveModel(model);
    setCurrentTab('runs');
  };

  const handleExperimentComplete = (manifest: ExperimentManifest) => {
    setCurrentManifest(manifest);
    setRunsCount(prev => prev + 1);
  };

  const handlePromoteToRelease = (modelId: string) => {
    setSelectedReleaseModelId(modelId);
    setCurrentTab('releases');
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100vh',
      width: '100vw',
      backgroundColor: 'var(--bg-app)',
      color: 'var(--text-main)',
      overflow: 'hidden'
    }}>
      <Header
        currentTab={currentTab}
        onNavigate={setCurrentTab}
        backendOnline={backendOnline}
        healthStatus={healthStatus}
        activeModelName={activeModel?.name}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar
          currentTab={currentTab}
          onSelectTab={setCurrentTab}
          modelsCount={modelsCount}
          runsCount={runsCount}
          resultsAvailable={!!currentManifest}
          releasesCount={modelsCount}
        />

        <main style={{
          flex: 1,
          overflowY: 'auto',
          backgroundColor: 'var(--bg-app)',
          height: 'calc(100vh - 56px)'
        }}>
          {currentTab === 'models' && (
            <ModelsWorkspaceView
              onSelectModelForRun={handleSelectModelForRun}
              onNavigateTab={setCurrentTab}
            />
          )}

          {currentTab === 'runs' && (
            <RunsWorkspaceView
              activeModel={activeModel}
              onSelectModel={setActiveModel}
              onNavigateTab={setCurrentTab}
              onExperimentComplete={handleExperimentComplete}
            />
          )}

          {currentTab === 'results' && (
            <ResultsWorkspaceView
              manifest={currentManifest}
              onNavigateTab={setCurrentTab}
              onPromoteToRelease={handlePromoteToRelease}
            />
          )}

          {currentTab === 'releases' && (
            <ReleasesWorkspaceView
              initialModelId={selectedReleaseModelId}
              onNavigateTab={setCurrentTab}
            />
          )}
        </main>
      </div>
    </div>
  );
};
