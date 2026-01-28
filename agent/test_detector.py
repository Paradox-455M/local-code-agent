"""Detect test framework and configuration in a project."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class TestFrameworkInfo:
    """Information about detected test framework."""
    
    name: str  # pytest, unittest, jest, go test, etc.
    config_file: Optional[Path] = None
    test_command: Optional[str] = None
    test_patterns: List[str] = None
    coverage_available: bool = False
    
    def __post_init__(self):
        if self.test_patterns is None:
            self.test_patterns = []


class TestFrameworkDetector:
    """Detect test framework used in a project."""
    
    def __init__(self, repo_root: Path = None):
        """
        Initialize test framework detector.
        
        Args:
            repo_root: Root directory of the repository. Defaults to current directory.
        """
        self.repo_root = repo_root or Path.cwd()
    
    def detect(self) -> TestFrameworkInfo:
        """
        Detect the test framework used in the project.
        
        Returns:
            TestFrameworkInfo with detected information.
        """
        # Check for Python test frameworks
        if self._has_pytest():
            return self._detect_pytest()
        elif self._has_unittest():
            return self._detect_unittest()
        
        # Check for JavaScript test frameworks
        if self._has_jest():
            return self._detect_jest()
        elif self._has_mocha():
            return self._detect_mocha()
        
        # Check for Go tests
        if self._has_go_tests():
            return self._detect_go_test()
        
        # Check for Rust tests
        if self._has_rust_tests():
            return self._detect_cargo_test()
        
        # Default fallback
        return TestFrameworkInfo(
            name="unknown",
            test_command="",
            test_patterns=[]
        )
    
    def _has_pytest(self) -> bool:
        """Check if pytest is available."""
        # Check for pytest.ini, pyproject.toml with pytest config, or pytest in requirements
        pytest_ini = self.repo_root / "pytest.ini"
        pyproject_toml = self.repo_root / "pyproject.toml"
        setup_cfg = self.repo_root / "setup.cfg"
        requirements_files = [
            self.repo_root / "requirements.txt",
            self.repo_root / "requirements-dev.txt",
            self.repo_root / "dev-requirements.txt",
        ]
        
        if pytest_ini.exists():
            return True
        
        if pyproject_toml.exists():
            content = pyproject_toml.read_text()
            if "[tool.pytest" in content:
                return True
        
        if setup_cfg.exists():
            content = setup_cfg.read_text()
            if "[tool:pytest]" in content or "[pytest]" in content:
                return True
        
        for req_file in requirements_files:
            if req_file.exists():
                content = req_file.read_text()
                if re.search(r'^pytest[>=<]', content, re.MULTILINE):
                    return True
        
        # Check for test files that import pytest
        test_dirs = ["tests", "test"]
        for test_dir in test_dirs:
            test_path = self.repo_root / test_dir
            if test_path.exists():
                for test_file in test_path.glob("test_*.py"):
                    content = test_file.read_text()
                    if "import pytest" in content or "from pytest" in content:
                        return True
        
        return False
    
    def _has_unittest(self) -> bool:
        """Check if unittest is being used."""
        test_dirs = ["tests", "test"]
        for test_dir in test_dirs:
            test_path = self.repo_root / test_dir
            if test_path.exists():
                for test_file in test_path.glob("test_*.py"):
                    content = test_file.read_text()
                    if "import unittest" in content or "from unittest" in content:
                        return True
        return False
    
    def _has_jest(self) -> bool:
        """Check if Jest is available."""
        package_json = self.repo_root / "package.json"
        if not package_json.exists():
            return False
        
        try:
            import json
            content = json.loads(package_json.read_text())
            dev_deps = content.get("devDependencies", {})
            deps = content.get("dependencies", {})
            
            return "jest" in dev_deps or "jest" in deps
        except Exception:
            return False
    
    def _has_mocha(self) -> bool:
        """Check if Mocha is available."""
        package_json = self.repo_root / "package.json"
        if not package_json.exists():
            return False
        
        try:
            import json
            content = json.loads(package_json.read_text())
            dev_deps = content.get("devDependencies", {})
            
            return "mocha" in dev_deps
        except Exception:
            return False
    
    def _has_go_tests(self) -> bool:
        """Check if Go tests exist."""
        return any((self.repo_root / "go.mod").exists() or 
                   list(self.repo_root.glob("*_test.go")))
    
    def _has_rust_tests(self) -> bool:
        """Check if Rust tests exist."""
        return (self.repo_root / "Cargo.toml").exists()
    
    def _detect_pytest(self) -> TestFrameworkInfo:
        """Detect pytest configuration."""
        config_file = None
        test_patterns = ["test_*.py", "*_test.py"]
        
        # Check for config files
        pytest_ini = self.repo_root / "pytest.ini"
        pyproject_toml = self.repo_root / "pyproject.toml"
        setup_cfg = self.repo_root / "setup.cfg"
        
        if pytest_ini.exists():
            config_file = pytest_ini
        elif pyproject_toml.exists() and "[tool.pytest" in pyproject_toml.read_text():
            config_file = pyproject_toml
        elif setup_cfg.exists():
            config_file = setup_cfg
        
        # Build test command
        test_command = "pytest"
        
        # Check if pytest-cov is available
        coverage_available = False
        requirements_files = [
            self.repo_root / "requirements.txt",
            self.repo_root / "requirements-dev.txt",
            self.repo_root / "dev-requirements.txt",
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                content = req_file.read_text()
                if "pytest-cov" in content or "coverage" in content:
                    coverage_available = True
                    test_command = "pytest --cov"
                    break
        
        return TestFrameworkInfo(
            name="pytest",
            config_file=config_file,
            test_command=test_command,
            test_patterns=test_patterns,
            coverage_available=coverage_available
        )
    
    def _detect_unittest(self) -> TestFrameworkInfo:
        """Detect unittest configuration."""
        return TestFrameworkInfo(
            name="unittest",
            test_command="python -m unittest discover",
            test_patterns=["test_*.py"],
            coverage_available=False
        )
    
    def _detect_jest(self) -> TestFrameworkInfo:
        """Detect Jest configuration."""
        jest_config = self.repo_root / "jest.config.js"
        
        return TestFrameworkInfo(
            name="jest",
            config_file=jest_config if jest_config.exists() else None,
            test_command="npm test",
            test_patterns=["*.test.js", "*.spec.js"],
            coverage_available=True
        )
    
    def _detect_mocha(self) -> TestFrameworkInfo:
        """Detect Mocha configuration."""
        return TestFrameworkInfo(
            name="mocha",
            test_command="npm test",
            test_patterns=["*.test.js", "*.spec.js"],
            coverage_available=False
        )
    
    def _detect_go_test(self) -> TestFrameworkInfo:
        """Detect Go test configuration."""
        return TestFrameworkInfo(
            name="go test",
            test_command="go test ./...",
            test_patterns=["*_test.go"],
            coverage_available=True
        )
    
    def _detect_cargo_test(self) -> TestFrameworkInfo:
        """Detect Cargo (Rust) test configuration."""
        return TestFrameworkInfo(
            name="cargo test",
            test_command="cargo test",
            test_patterns=["tests/*.rs"],
            coverage_available=False
        )
    
    def get_test_command(self, specific_file: Optional[str] = None) -> str:
        """
        Get the appropriate test command for the project.
        
        Args:
            specific_file: Optional specific test file to run.
        
        Returns:
            Test command string.
        """
        framework = self.detect()
        
        if specific_file:
            if framework.name == "pytest":
                return f"pytest {specific_file}"
            elif framework.name == "unittest":
                return f"python -m unittest {specific_file}"
            elif framework.name == "jest":
                return f"npm test -- {specific_file}"
            elif framework.name == "go test":
                return f"go test {specific_file}"
            elif framework.name == "cargo test":
                return f"cargo test --test {specific_file}"
        
        return framework.test_command or "pytest"
    
    def find_test_files(self) -> List[Path]:
        """
        Find all test files in the project.
        
        Returns:
            List of test file paths.
        """
        framework = self.detect()
        test_files = []
        
        # Common test directories
        test_dirs = ["tests", "test", "src", "."]
        
        for test_dir in test_dirs:
            test_path = self.repo_root / test_dir
            if not test_path.exists():
                continue
            
            for pattern in framework.test_patterns:
                test_files.extend(test_path.glob(f"**/{pattern}"))
        
        return sorted(set(test_files))
    
    def print_detection_info(self) -> None:
        """Print detected test framework information."""
        framework = self.detect()
        
        console.print(f"\n[bold]Test Framework Detection:[/bold]")
        console.print(f"  Framework: [cyan]{framework.name}[/cyan]")
        
        if framework.config_file:
            console.print(f"  Config: [dim]{framework.config_file.relative_to(self.repo_root)}[/dim]")
        
        console.print(f"  Command: [green]{framework.test_command}[/green]")
        
        if framework.test_patterns:
            patterns = ", ".join(framework.test_patterns)
            console.print(f"  Patterns: [dim]{patterns}[/dim]")
        
        if framework.coverage_available:
            console.print(f"  Coverage: [green]Available[/green]")
        
        test_files = self.find_test_files()
        console.print(f"  Test files found: [yellow]{len(test_files)}[/yellow]\n")


def detect_test_framework(repo_root: Path = None) -> TestFrameworkInfo:
    """
    Convenience function to detect test framework.
    
    Args:
        repo_root: Root directory of the repository.
    
    Returns:
        TestFrameworkInfo with detected information.
    """
    detector = TestFrameworkDetector(repo_root)
    return detector.detect()
