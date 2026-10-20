package ai.tinyai.controlplane.repository;

import ai.tinyai.controlplane.domain.Project;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ProjectRepository extends JpaRepository<Project, String> {
    List<Project> findAllByOrderByCreatedAtDesc();
}
