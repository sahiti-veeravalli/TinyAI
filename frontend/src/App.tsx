import React, { useState, useEffect } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { Header } from './components/Header';
import { OverviewView } from './components/OverviewView';
import { ProjectsView } from './components/ProjectsView';
import { DatasetsView } from './components/DatasetsView';
import { BaselinesView } from './components/BaselinesView';
import { OptimizationLabView } from './components/OptimizationLabView';
import { ModelRegistryView } from './components/ModelRegistryView';
import { DeploymentsSandboxView } from './components/DeploymentsSandboxView';
import { MonitoringView } from './components/MonitoringView';
import { ApiSdkView } from './components/ApiSdkView';
import { DocumentationView } from './components/DocumentationView';
import { checkBackendHealth, fetchOverview } from './api';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [overviewData, setOverviewData] = useState<any>(null);

  // Cross-page state transfers
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [selectedBaselineFamily, setSelectedBaselineFamily] = useState<string>('lightgbm');
  const [selectedTargetCol, setSelectedTargetCol] = useState<string>('is_fraud');

  useEffect(() => {
    const syncStatus = async () => {
      const isOnline = await checkBackendHealth();
      setBackendOnline(isOnline);
      if (isOnline) {
        try {
          const ov = await fetchOverview();
          setOverviewData(ov);
        } catch (e) {
          console.error(e);
        }
      }
    };
    syncStatus();
    const interval = setInterval(syncStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleDatasetSelectForBaseline = (datasetId: string, targetCol: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedTargetCol(targetCol);
    setCurrentTab('baselines');
  };

  const handleBaselineSelectForOptimization = (datasetId: string, modelFamily: string, targetCol: string) => {
    setSelectedDatasetId(datasetId);
    setSelectedBaselineFamily(modelFamily);
    setSelectedTargetCol(targetCol);
    setCurrentTab('optimization_lab');
  };

  const handlePromoteToRegistry = () => {
    setCurrentTab('model_registry');
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', backgroundColor: '#080c14' }}>
      {/* Persistent Navigation Sidebar */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        backendOnline={backendOnline}
      />

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
        <Header
          activeProjectName="Point-of-Sale Fraud Sentinel"
          onOpenDocs={() => setCurrentTab('documentation')}
          onLaunchQuickExp={() => setCurrentTab('optimization_lab')}
        />

        <main style={{ flex: 1, overflowY: 'auto', backgroundColor: '#080c14' }}>
          {currentTab === 'overview' && (
            <OverviewView
              onNavigate={setCurrentTab}
              overviewData={overviewData}
              backendOnline={backendOnline}
            />
          )}

          {currentTab === 'projects' && (
            <ProjectsView
              onNavigate={setCurrentTab}
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
        </main>
      </div>
    </div>
  );
};
