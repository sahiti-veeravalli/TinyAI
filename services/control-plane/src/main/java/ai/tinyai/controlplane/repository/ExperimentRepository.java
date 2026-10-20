package ai.tinyai.controlplane.repository;

import ai.tinyai.controlplane.domain.Experiment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ExperimentRepository extends JpaRepository<Experiment, String> {
    List<Experiment> findByProjectIdOrderByCreatedAtDesc(String projectId);
}
