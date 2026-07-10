"""
File system monitoring for project completion detection.
Watches project folders for marker files indicating project completion.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ProjectMonitor:
    """Monitors project folders for completion markers."""
    
    # Project completion markers
    COMPLETION_MARKERS = [
        ".deployment-ready",
        ".project-complete",
        "deployment.json",
        ".deploy",
    ]
    
    # Ignore patterns
    IGNORE_PATTERNS = [
        ".git",
        ".gitignore",
        "__pycache__",
        ".env",
        ".pytest_cache",
        ".venv",
        "venv",
        "node_modules",
    ]
    
    def __init__(self, monitor_path: str = "./projects"):
        self.monitor_path = Path(monitor_path)
        self.monitor_path.mkdir(parents=True, exist_ok=True)
        self.processed_projects = self._load_processed_projects()
    
    def _load_processed_projects(self) -> Dict:
        """Load list of already processed projects."""
        processed_file = self.monitor_path / ".processed_projects.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load processed projects: {e}")
        return {}
    
    def _save_processed_projects(self):
        """Save processed projects list."""
        processed_file = self.monitor_path / ".processed_projects.json"
        try:
            with open(processed_file, 'w') as f:
                json.dump(self.processed_projects, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processed projects: {e}")
    
    def _is_project_directory(self, path: Path) -> bool:
        """Check if directory is a valid project directory."""
        if not path.is_dir():
            return False
        
        # Skip ignored patterns
        for pattern in self.IGNORE_PATTERNS:
            if pattern in path.name:
                return False
        
        # Check for project files
        project_files = ['streamlit_app.py', 'app.py', 'main.py', 'package.json', 'index.py']
        return any((path / f).exists() for f in project_files)
    
    def _is_project_complete(self, project_path: Path) -> bool:
        """Check if project has completion marker."""
        for marker in self.COMPLETION_MARKERS:
            if (project_path / marker).exists():
                logger.info(f"Found completion marker: {marker} in {project_path.name}")
                return True
        return False
    
    def _read_deployment_config(self, project_path: Path) -> Dict:
        """Read deployment configuration from project."""
        config_file = project_path / "deployment.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read deployment config: {e}")
        return {}
    
    def scan_for_projects(self) -> List[Dict]:
        """Scan monitor folder for new projects."""
        projects = []
        
        if not self.monitor_path.exists():
            logger.warning(f"Monitor path does not exist: {self.monitor_path}")
            return projects
        
        try:
            for item in self.monitor_path.iterdir():
                if not self._is_project_directory(item):
                    continue
                
                project_name = item.name
                
                # Skip already processed projects
                if project_name in self.processed_projects:
                    logger.debug(f"Skipping already processed project: {project_name}")
                    continue
                
                # Check if project is complete
                if not self._is_project_complete(item):
                    logger.debug(f"Project not marked complete: {project_name}")
                    continue
                
                logger.info(f"Found completed project: {project_name}")
                
                deployment_config = self._read_deployment_config(item)
                
                projects.append({
                    "name": project_name,
                    "path": str(item.absolute()),
                    "detected_at": datetime.now().isoformat(),
                    "config": deployment_config,
                })
        
        except Exception as e:
            logger.error(f"Error scanning projects: {e}")
        
        return projects
    
    def mark_as_processed(self, project_name: str, deployment_info: Dict = None):
        """Mark project as processed."""
        self.processed_projects[project_name] = {
            "processed_at": datetime.now().isoformat(),
            "deployment_info": deployment_info or {},
        }
        self._save_processed_projects()
        logger.info(f"Marked project as processed: {project_name}")
    
    def get_project_status(self) -> Dict:
        """Get status of all projects."""
        return {
            "total_processed": len(self.processed_projects),
            "processed_projects": self.processed_projects,
            "monitor_path": str(self.monitor_path.absolute()),
        }
