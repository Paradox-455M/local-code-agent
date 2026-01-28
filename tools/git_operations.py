"""Git operations integration."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class GitStatus:
    """Git repository status."""
    
    is_repo: bool
    branch: Optional[str] = None
    has_changes: bool = False
    untracked_files: List[str] = None
    modified_files: List[str] = None
    
    def __post_init__(self):
        if self.untracked_files is None:
            self.untracked_files = []
        if self.modified_files is None:
            self.modified_files = []


def get_git_status(repo_root: Path) -> GitStatus:
    """
    Get git repository status.
    
    Args:
        repo_root: Repository root directory.
    
    Returns:
        GitStatus object.
    """
    if not (repo_root / ".git").exists():
        return GitStatus(is_repo=False)
    
    try:
        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        branch = branch_result.stdout.strip()
        
        # Check for changes
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        
        has_changes = bool(status_result.stdout.strip())
        modified_files = []
        untracked_files = []
        
        for line in status_result.stdout.splitlines():
            if not line.strip():
                continue
            status = line[:2]
            filename = line[3:]
            
            if status[0] == "?":
                untracked_files.append(filename)
            elif status[0] in ["M", "A", "D", "R"]:
                modified_files.append(filename)
        
        return GitStatus(
            is_repo=True,
            branch=branch,
            has_changes=has_changes,
            modified_files=modified_files,
            untracked_files=untracked_files,
        )
    except Exception:
        return GitStatus(is_repo=True)


def create_branch(repo_root: Path, branch_name: str) -> bool:
    """
    Create a new git branch.
    
    Args:
        repo_root: Repository root directory.
        branch_name: Name of the branch to create.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


def generate_commit_message(task: str, files: List[str]) -> str:
    """
    Generate a commit message from task and files.
    
    Args:
        task: The task description.
        files: List of modified files.
    
    Returns:
        Commit message string.
    """
    # Simple heuristic-based commit message
    task_lower = task.lower()
    
    # Determine commit type
    if any(kw in task_lower for kw in ["fix", "bug", "error", "issue"]):
        prefix = "fix"
    elif any(kw in task_lower for kw in ["add", "create", "implement", "new"]):
        prefix = "feat"
    elif any(kw in task_lower for kw in ["refactor", "restructure", "reorganize"]):
        prefix = "refactor"
    elif any(kw in task_lower for kw in ["update", "improve", "enhance"]):
        prefix = "chore"
    else:
        prefix = "chore"
    
    # Create summary
    summary = task[:72]  # Limit to 72 chars
    if len(task) > 72:
        summary = summary.rsplit(" ", 1)[0] + "..."
    
    # Add file list if small
    if len(files) <= 3:
        file_list = ", ".join(files)
        if len(summary) + len(file_list) + 2 <= 72:
            summary = f"{summary} ({file_list})"
    
    return f"{prefix}: {summary}"


def commit_changes(
    repo_root: Path,
    message: Optional[str] = None,
    files: Optional[List[str]] = None,
    auto_generate_message: bool = True,
    task: Optional[str] = None,
) -> bool:
    """
    Commit changes to git repository.
    
    Args:
        repo_root: Repository root directory.
        message: Optional commit message.
        files: Optional list of files to commit (if None, commits all changes).
        auto_generate_message: Whether to auto-generate message if not provided.
        task: Task description for auto-generated message.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Stage files
        if files:
            for file in files:
                subprocess.run(
                    ["git", "add", file],
                    cwd=str(repo_root),
                    check=True,
                    capture_output=True,
                )
        else:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(repo_root),
                check=True,
                capture_output=True,
            )
        
        # Generate message if needed
        if not message and auto_generate_message:
            if task:
                staged_files = files or []
                message = generate_commit_message(task, staged_files)
            else:
                message = "chore: update code"
        
        if not message:
            return False
        
        # Commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Demo
    repo_root = Path(".")
    status = get_git_status(repo_root)
    print(f"Repository: {status.is_repo}")
    if status.is_repo:
        print(f"Branch: {status.branch}")
        print(f"Has changes: {status.has_changes}")
        print(f"Modified: {status.modified_files}")
        print(f"Untracked: {status.untracked_files}")
