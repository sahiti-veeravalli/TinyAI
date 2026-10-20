import React, { useState, useEffect } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { Header } from './components/Header';
import { ExperimentWorkflowView } from './components/ExperimentWorkflowView';
import { ToolsHubView } from './components/ToolsHubView';
import { ToolWorkspaceView } from './components/ToolWorkspaceView';
import { ProjectsView } from './components/ProjectsView';
import { DatasetsView } from './components/DatasetsView';
import { BaselinesView } from './components/BaselinesView';
import { OptimizationLabView } from './components/OptimizationLabView';
import { ModelRegistryView } from './components/ModelRegistryView';
import { DeploymentsSandboxView } from './components/DeploymentsSandboxView';
import { MonitoringView } from './components/MonitoringView';
import { ApiSdkView } from './components/ApiSdkView';
import { DocumentationView } from './components/DocumentationView';
import { fetchHealthDetails, fetchOverview, HealthStatus } from './api';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('experiment');
  const [activeToolId, setActiveToolId] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [overviewData, setOverviewData] = useState<any>(null);

  // Cross-page state transfers
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [selectedBaselineFamily, setSelectedBaselineFamily] = useState<string>('lightgbm');
  const [selectedTargetCol, setSelectedTargetCol] = useState<string>('is_fraud');

  useEffect(() => {
    const syncStatus = async () => {
      const details = await fetchHealthDetails();
      setHealthStatus(details);
      setBackendOnline(!!details);
      if (details) {
        try {
          const ov = await fetchOverview();
          setOverviewData(ov);
        } catch (e) {
          console.error(e);
        }
      }
    };
    syncStatus();
    const interval = setInterval(syncStatus, 8000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectTool = (toolId: string) => {
    setActiveToolId(toolId);
  };

  const handleBackFromTool = () => {
    setActiveToolId(null);
  };

  const handleNavigateTab = (tab: NavTab) => {
    setActiveToolId(null);
    setCurrentTab(tab);
  };

  const handleDatasetSelectForBaseline = (datasetId: string, targetCol: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedTargetCol(targetCol);
    setActiveToolId(null);
    setCurrentTab('baselines');
  };

  const handleBaselineSelectForOptimization = (datasetId: string, modelFamily: string, targetCol: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedBaselineFamily(modelFamily);
    setSelectedTargetCol(targetCol);
    setActiveToolId(null);
    setCurrentTab('optimization_lab');
  };

  const handlePromoteToRegistry = () => {
    setActiveToolId(null);
    setCurrentTab('model_registry');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', width: '100vw', backgroundColor: '#f8fafc', color: '#0f172a' }}>
      <Header
        currentTab={currentTab}
        onNavigate={handleNavigateTab}
        onSelectTool={handleSelectTool}
        backendOnline={backendOnline}
        healthStatus={healthStatus}
      />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar
          currentTab={currentTab}
          onSelectTab={handleNavigateTab}
          backendOnline={backendOnline}
        />

        <main style={{ flex: 1, overflowY: 'auto', backgroundColor: '#f8fafc', height: 'calc(100vh - 64px)' }}>
          {activeToolId ? (
            <ToolWorkspaceView
              toolId={activeToolId}
              onBack={handleBackFromTool}
              onNavigateTab={handleNavigateTab}
            />
          ) : (
            <>
              {currentTab === 'experiment' && (
                <ExperimentWorkflowView
                  onNavigateTab={handleNavigateTab}
                />
              )}

              {(currentTab === 'tools_hub' || currentTab === 'overview') && (
                <ToolsHubView
                  onSelectTool={handleSelectTool}
                  onNavigateTab={handleNavigateTab}
                />
              )}

              {currentTab === 'projects' && (
                <ProjectsView
                  onNavigate={handleNavigateTab}
                  projects={overviewData?.projects || []}
                />
              )}

              {currentTab === 'datasets' && (
                <DatasetsView
                  onDatasetSelect={handleDatasetSelectForBaseline}
                />
              )}

              {currentTab === 'baselines' && (
                <BaselinesView
                  onSelectBaselineForOptimization={handleBaselineSelectForOptimization}
                />
              )}

              {currentTab === 'optimization_lab' && (
                <OptimizationLabView
                  initialDatasetId={selectedDatasetId}
                  initialBaselineFamily={selectedBaselineFamily}
                  initialTargetCol={selectedTargetCol}
                  onPromoteToRegistry={handlePromoteToRegistry}
                />
              )}

              {currentTab === 'model_registry' && (
                <ModelRegistryView />
              )}

              {currentTab === 'deployments' && (
                <DeploymentsSandboxView />
              )}

              {currentTab === 'monitoring' && (
                <MonitoringView />
              )}

              {currentTab === 'api_sdk' && (
                <ApiSdkView />
              )}

              {currentTab === 'documentation' && (
                <DocumentationView />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
};
