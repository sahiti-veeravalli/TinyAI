package ai.tinyai.controlplane.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "experiments")
public class Experiment {

    @Id
    @Column(length = 64)
    private String id;

    @Column(name = "project_id", nullable = false)
    private String projectId;

    @Column(name = "dataset_id", nullable = false)
    private String datasetId;

    @Column(name = "baseline_family", nullable = false)
    private String baselineFamily;

    @Column(nullable = false)
    private String status;

    @Column(nullable = false)
    private String priority;

    @Column(name = "max_disk_size_kb")
    private Double maxDiskSizeKb;

    @Column(name = "max_latency_ms")
    private Double maxLatencyMs;

    @Column(name = "min_quality_score")
    private Double minQualityScore;

    @Column(name = "created_at")
    private Instant createdAt = Instant.now();

    public Experiment() {}

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getProjectId() { return projectId; }
    public void setProjectId(String projectId) { this.projectId = projectId; }

    public String getDatasetId() { return datasetId; }
    public void setDatasetId(String datasetId) { this.datasetId = datasetId; }

    public String getBaselineFamily() { return baselineFamily; }
    public void setBaselineFamily(String baselineFamily) { this.baselineFamily = baselineFamily; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getPriority() { return priority; }
    public void setPriority(String priority) { this.priority = priority; }

    public Double getMaxDiskSizeKb() { return maxDiskSizeKb; }
    public void setMaxDiskSizeKb(Double maxDiskSizeKb) { this.maxDiskSizeKb = maxDiskSizeKb; }

    public Double getMaxLatencyMs() { return maxLatencyMs; }
    public void setMaxLatencyMs(Double maxLatencyMs) { this.maxLatencyMs = maxLatencyMs; }

    public Double getMinQualityScore() { return minQualityScore; }
    public void setMinQualityScore(Double minQualityScore) { this.minQualityScore = minQualityScore; }

    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
