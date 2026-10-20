package ai.tinyai.controlplane.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "model_artifacts")
public class ModelArtifact {

    @Id
    @Column(length = 64)
    private String id;

    @Column(name = "experiment_id", nullable = false)
    private String experimentId;

    @Column(name = "project_id", nullable = false)
    private String projectId;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private String version;

    @Column(nullable = false)
    private String strategy;

    @Column(name = "model_family", nullable = false)
    private String modelFamily;

    @Column(name = "quality_score", nullable = false)
    private Double qualityScore;

    @Column(name = "latency_p50_ms", nullable = false)
    private Double latencyP50Ms;

    @Column(name = "disk_size_kb", nullable = false)
    private Double diskSizeKb;

    @Column(name = "compression_ratio", nullable = false)
    private Double compressionRatio;

    @Column(name = "artifact_path")
    private String artifactPath;

    @Column(nullable = false)
    private String status;

    @Column(name = "created_at")
    private Instant createdAt = Instant.now();

    public ModelArtifact() {}

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getExperimentId() { return experimentId; }
    public void setExperimentId(String experimentId) { this.experimentId = experimentId; }

    public String getProjectId() { return projectId; }
    public void setProjectId(String projectId) { this.projectId = projectId; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }

    public String getStrategy() { return strategy; }
    public void setStrategy(String strategy) { this.strategy = strategy; }

    public String getModelFamily() { return modelFamily; }
    public void setModelFamily(String modelFamily) { this.modelFamily = modelFamily; }

    public Double getQualityScore() { return qualityScore; }
    public void setQualityScore(Double qualityScore) { this.qualityScore = qualityScore; }

    public Double getLatencyP50Ms() { return latencyP50Ms; }
    public void setLatencyP50Ms(Double latencyP50Ms) { this.latencyP50Ms = latencyP50Ms; }

    public Double getDiskSizeKb() { return diskSizeKb; }
    public void setDiskSizeKb(Double diskSizeKb) { this.diskSizeKb = diskSizeKb; }

    public Double getCompressionRatio() { return compressionRatio; }
    public void setCompressionRatio(Double compressionRatio) { this.compressionRatio = compressionRatio; }

    public String getArtifactPath() { return artifactPath; }
    public void setArtifactPath(String artifactPath) { this.artifactPath = artifactPath; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
