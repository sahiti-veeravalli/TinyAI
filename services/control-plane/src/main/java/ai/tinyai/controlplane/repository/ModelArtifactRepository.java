package ai.tinyai.controlplane.repository;

import ai.tinyai.controlplane.domain.ModelArtifact;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ModelArtifactRepository extends JpaRepository<ModelArtifact, String> {
    List<ModelArtifact> findByProjectIdOrderByCreatedAtDesc(String projectId);
    List<ModelArtifact> findByExperimentId(String experimentId);
}
