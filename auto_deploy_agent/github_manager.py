"""
GitHub integration for automatic repository management.
Handles pushing code, creating commits, and managing branches.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class GitHubManager:
    """Manages GitHub operations for project deployment."""
    
    def __init__(self, token: str, username: str, email: str):
        self.token = token
        self.username = username
        self.email = email
        self.git_installed = self._check_git()
    
    def _check_git(self) -> bool:
        """Check if git is installed."""
        try:
            subprocess.run(
                ["git", "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            return True
        except Exception as e:
            logger.error(f"Git not installed or not available: {e}")
            return False
    
    def _run_git_command(
        self, 
        command: list, 
        cwd: Optional[str] = None,
        timeout: int = 30
    ) -> Tuple[bool, str, str]:
        """Run a git command and return result."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "GIT_AUTHOR_EMAIL": self.email, "GIT_AUTHOR_NAME": self.username}
            )
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            return False, e.stdout, e.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out")
            return False, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error running git command: {e}")
            return False, "", str(e)
    
    def initialize_repository(self, project_path: str) -> bool:
        """Initialize git repository if not already initialized."""
        if not self.git_installed:
            logger.error("Git is not installed")
            return False
        
        project_path = Path(project_path)
        
        # Check if already initialized
        if (project_path / ".git").exists():
            logger.info(f"Repository already initialized: {project_path}")
            return True
        
        # Initialize
        success, stdout, stderr = self._run_git_command(
            ["git", "init"],
            cwd=str(project_path)
        )
        
        if success:
            logger.info(f"Initialized git repository: {project_path}")
        else:
            logger.error(f"Failed to initialize repository: {stderr}")
        
        return success
    
    def configure_repository(self, project_path: str) -> bool:
        """Configure git user for repository."""
        if not self.git_installed:
            return False
        
        commands = [
            ["git", "config", "user.email", self.email],
            ["git", "config", "user.name", self.username],
        ]
        
        for command in commands:
            success, _, stderr = self._run_git_command(
                command,
                cwd=project_path
            )
            if not success:
                logger.error(f"Failed to configure git: {stderr}")
                return False
        
        logger.info(f"Configured git repository: {project_path}")
        return True
    
    def add_remote(self, project_path: str, remote_url: str, remote_name: str = "origin") -> bool:
        """Add remote repository."""
        if not self.git_installed:
            return False
        
        # Check if remote already exists
        success, stdout, _ = self._run_git_command(
            ["git", "remote", "get-url", remote_name],
            cwd=project_path
        )
        
        if success and stdout.strip() == remote_url:
            logger.info(f"Remote already configured: {remote_name}")
            return True
        
        # Remove existing remote if it points to different URL
        if success:
            self._run_git_command(
                ["git", "remote", "remove", remote_name],
                cwd=project_path
            )
        
        # Add remote
        success, _, stderr = self._run_git_command(
            ["git", "remote", "add", remote_name, remote_url],
            cwd=project_path
        )
        
        if success:
            logger.info(f"Added remote: {remote_name} -> {remote_url}")
        else:
            logger.error(f"Failed to add remote: {stderr}")
        
        return success
    
    def add_all_changes(self, project_path: str) -> bool:
        """Stage all changes for commit."""
        if not self.git_installed:
            return False
        
        success, _, stderr = self._run_git_command(
            ["git", "add", "-A"],
            cwd=project_path
        )
        
        if success:
            logger.info("Staged all changes")
        else:
            logger.error(f"Failed to stage changes: {stderr}")
        
        return success
    
    def commit_changes(self, project_path: str, message: str) -> bool:
        """Commit changes to repository."""
        if not self.git_installed:
            return False
        
        success, stdout, stderr = self._run_git_command(
            ["git", "commit", "-m", message],
            cwd=project_path
        )
        
        # 'nothing to commit' is not an error
        if not success and "nothing to commit" not in stderr.lower():
            logger.error(f"Failed to commit: {stderr}")
            return False
        
        logger.info(f"Committed changes: {message}")
        return True
    
    def push_to_remote(self, project_path: str, branch: str = "main", remote: str = "origin") -> bool:
        """Push changes to remote repository."""
        if not self.git_installed:
            return False
        
        success, _, stderr = self._run_git_command(
            ["git", "push", "-u", remote, branch],
            cwd=project_path,
            timeout=60
        )
        
        if success:
            logger.info(f"Pushed to {remote}/{branch}")
        else:
            logger.error(f"Failed to push: {stderr}")
        
        return success
    
    def push_project_to_github(
        self, 
        project_path: str, 
        repo_name: str,
        commit_message: str = "Auto-deploy: Project ready for deployment"
    ) -> Tuple[bool, Optional[str]]:
        """
        Complete workflow: initialize, configure, add remote, commit, and push.
        Returns (success, remote_url)
        """
        project_path = Path(project_path)
        
        # Step 1: Initialize
        if not self.initialize_repository(str(project_path)):
            return False, None
        
        # Step 2: Configure
        if not self.configure_repository(str(project_path)):
            return False, None
        
        # Step 3: Add remote
        remote_url = f"https://{self.username}:{self.token}@github.com/{self.username}/{repo_name}.git"
        if not self.add_remote(str(project_path), remote_url):
            # Try HTTPS without credentials for public repos
            remote_url = f"https://github.com/{self.username}/{repo_name}.git"
            if not self.add_remote(str(project_path), remote_url):
                return False, None
        
        # Step 4: Commit
        if not self.add_all_changes(str(project_path)):
            return False, None
        
        if not self.commit_changes(str(project_path), commit_message):
            return False, None
        
        # Step 5: Push
        if not self.push_to_remote(str(project_path)):
            return False, None
        
        logger.info(f"Successfully pushed project to GitHub: {remote_url}")
        return True, remote_url
