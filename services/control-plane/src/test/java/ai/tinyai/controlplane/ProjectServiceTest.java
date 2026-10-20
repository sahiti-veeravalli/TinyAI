package ai.tinyai.controlplane;

import ai.tinyai.controlplane.domain.Project;
import ai.tinyai.controlplane.repository.ProjectRepository;
import ai.tinyai.controlplane.service.ProjectService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class ProjectServiceTest {

    @Autowired
    private ProjectService projectService;

    @Autowired
    private ProjectRepository projectRepository;

    @Test
    void testCreateAndRetrieveProject() {
        Project created = projectService.createProject(
                "Autonomous Drone Nav",
                "Sub-5ms visual odometry & obstacle classification.",
                "NVIDIA Jetson Nano"
        );

        assertNotNull(created.getId());
        assertTrue(created.getId().startsWith("proj_"));
        assertEquals("Autonomous Drone Nav", created.getName());

        List<Project> projects = projectService.getAllProjects();
        assertFalse(projects.isEmpty());
        assertTrue(projects.stream().anyMatch(p -> p.getId().equals(created.getId())));
    }
}
