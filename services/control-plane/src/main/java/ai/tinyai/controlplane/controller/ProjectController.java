package ai.tinyai.controlplane.controller;

import ai.tinyai.controlplane.domain.Project;
import ai.tinyai.controlplane.service.ProjectService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/projects")
@CrossOrigin(origins = "*")
public class ProjectController {

    private final ProjectService projectService;

    public ProjectController(ProjectService projectService) {
        this.projectService = projectService;
    }

    @GetMapping
    public ResponseEntity<List<Project>> listProjects() {
        return ResponseEntity.ok(projectService.getAllProjects());
    }

    @GetMapping("/{id}")
    public ResponseEntity<Project> getProject(@PathVariable String id) {
        return projectService.getProjectById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    public record CreateProjectRequest(String name, String description, String targetRuntime) {}

    @PostMapping
    public ResponseEntity<Project> createProject(@RequestBody CreateProjectRequest req) {
        Project created = projectService.createProject(req.name(), req.description(), req.targetRuntime());
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }
}
