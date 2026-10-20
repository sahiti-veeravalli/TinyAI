package ai.tinyai.controlplane.service;

import ai.tinyai.controlplane.domain.Project;
import ai.tinyai.controlplane.repository.ProjectRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
@Transactional
public class ProjectService {

    private final ProjectRepository projectRepository;

    public ProjectService(ProjectRepository projectRepository) {
        this.projectRepository = projectRepository;
    }

    @Transactional(readOnly = true)
    public List<Project> getAllProjects() {
        return projectRepository.findAllByOrderByCreatedAtDesc();
    }

    @Transactional(readOnly = true)
    public Optional<Project> getProjectById(String id) {
        return projectRepository.findById(id);
    }

    public Project createProject(String name, String description, String targetRuntime) {
        String id = "proj_" + UUID.randomUUID().toString().replace("-", "").substring(0, 8);
        Project project = new Project(id, name, description, targetRuntime);
        return projectRepository.save(project);
    }
}
