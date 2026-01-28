"""Test runner - Run tests and parse results intelligently."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess
import re


@dataclass
class TestResult:
    """Result of a test run."""
    
    passed: bool
    test_name: str
    duration: Optional[float] = None
    error: Optional[str] = None
    output: Optional[str] = None


@dataclass
class TestRunResult:
    """Result of running tests."""
    
    success: bool
    total_tests: int
    passed: int
    failed: int
    skipped: int = 0
    results: List[TestResult] = field(default_factory=list)
    output: str = ""
    error_output: str = ""
    framework: Optional[str] = None


class TestRunner:
    """Intelligent test runner."""
    
    def __init__(self, repo_root: Path, framework: Optional[str] = None):
        """
        Initialize test runner.
        
        Args:
            repo_root: Repository root directory.
            framework: Optional test framework ('pytest', 'unittest', 'jest', etc.).
        """
        self.repo_root = Path(repo_root).resolve()
        self.framework = framework or self._detect_framework()
    
    def _detect_framework(self) -> str:
        """Detect test framework from project."""
        # Check for pytest
        if (self.repo_root / "pytest.ini").exists() or (self.repo_root / "pyproject.toml").exists():
            try:
                pyproject = (self.repo_root / "pyproject.toml").read_text()
                if "pytest" in pyproject:
                    return "pytest"
            except Exception:
                pass
        
        # Check for requirements.txt
        req_file = self.repo_root / "requirements.txt"
        if req_file.exists():
            try:
                content = req_file.read_text()
                if "pytest" in content:
                    return "pytest"
                if "unittest" in content:
                    return "unittest"
            except Exception:
                pass
        
        # Check for package.json (JavaScript)
        package_json = self.repo_root / "package.json"
        if package_json.exists():
            try:
                import json
                pkg = json.loads(package_json.read_text())
                if "jest" in pkg.get("devDependencies", {}):
                    return "jest"
                if "mocha" in pkg.get("devDependencies", {}):
                    return "mocha"
            except Exception:
                pass
        
        # Default to pytest for Python
        return "pytest"
    
    def run_tests(
        self,
        test_path: Optional[Path] = None,
        test_name: Optional[str] = None,
        verbose: bool = False,
    ) -> TestRunResult:
        """
        Run tests.
        
        Args:
            test_path: Optional specific test file or directory.
            test_name: Optional specific test name to run.
            verbose: Whether to run in verbose mode.
        
        Returns:
            TestRunResult with test results.
        """
        if self.framework == "pytest":
            return self._run_pytest(test_path, test_name, verbose)
        elif self.framework == "unittest":
            return self._run_unittest(test_path, test_name, verbose)
        elif self.framework == "jest":
            return self._run_jest(test_path, test_name, verbose)
        else:
            return TestRunResult(
                success=False,
                total_tests=0,
                passed=0,
                failed=0,
                output="",
                error_output=f"Unknown framework: {self.framework}",
            )
    
    def _run_pytest(
        self,
        test_path: Optional[Path],
        test_name: Optional[str],
        verbose: bool,
    ) -> TestRunResult:
        """Run pytest tests."""
        cmd = ["pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if test_path:
            cmd.append(str(test_path))
        
        if test_name:
            cmd.extend(["-k", test_name])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            return self._parse_pytest_output(result.stdout, result.stderr, result.returncode == 0)
        except subprocess.TimeoutExpired:
            return TestRunResult(
                success=False,
                total_tests=0,
                passed=0,
                failed=0,
                output="",
                error_output="Test run timed out",
            )
        except Exception as e:
            return TestRunResult(
                success=False,
                total_tests=0,
                passed=0,
                failed=0,
                output="",
                error_output=str(e),
            )
    
    def _run_unittest(
        self,
        test_path: Optional[Path],
        test_name: Optional[str],
        verbose: bool,
    ) -> TestRunResult:
        """Run unittest tests."""
        cmd = ["python", "-m", "unittest"]
        
        if verbose:
            cmd.append("-v")
        
        if test_path:
            cmd.append(str(test_path))
        
        if test_name:
            cmd.extend(["-k", test_name])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            return self._parse_unittest_output(result.stdout, result.stderr, result.returncode == 0)
        except Exception as e:
            return TestRunResult(
                success=False,
                total_tests=0,
                passed=0,
                failed=0,
                output="",
                error_output=str(e),
            )
    
    def _run_jest(
        self,
        test_path: Optional[Path],
        test_name: Optional[str],
        verbose: bool,
    ) -> TestRunResult:
        """Run Jest tests."""
        cmd = ["npm", "test", "--"]
        
        if test_path:
            cmd.append(str(test_path))
        
        if test_name:
            cmd.extend(["-t", test_name])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            return self._parse_jest_output(result.stdout, result.stderr, result.returncode == 0)
        except Exception as e:
            return TestRunResult(
                success=False,
                total_tests=0,
                passed=0,
                failed=0,
                output="",
                error_output=str(e),
            )
    
    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        success: bool,
    ) -> TestRunResult:
        """Parse pytest output."""
        results: List[TestResult] = []
        
        # Extract test results
        # Pattern: test_file.py::test_function PASSED
        test_pattern = r"(\S+)::(\S+)\s+(PASSED|FAILED|SKIPPED)"
        matches = re.findall(test_pattern, stdout)
        
        passed = 0
        failed = 0
        skipped = 0
        
        for file_path, test_name, status in matches:
            test_result = TestResult(
                passed=(status == "PASSED"),
                test_name=f"{file_path}::{test_name}",
            )
            
            if status == "PASSED":
                passed += 1
            elif status == "FAILED":
                failed += 1
            else:
                skipped += 1
            
            results.append(test_result)
        
        # Extract summary
        summary_pattern = r"(\d+)\s+(passed|failed|skipped)"
        summary_matches = re.findall(summary_pattern, stdout)
        
        total = passed + failed + skipped
        
        return TestRunResult(
            success=success,
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            output=stdout,
            error_output=stderr,
            framework="pytest",
        )
    
    def _parse_unittest_output(
        self,
        stdout: str,
        stderr: str,
        success: bool,
    ) -> TestRunResult:
        """Parse unittest output."""
        results: List[TestResult] = []
        
        # Extract test results
        # Pattern: test_function (module.TestClass) ... ok
        test_pattern = r"(\S+)\s+\((\S+)\)\s+\.\.\.\s+(ok|FAIL|SKIP)"
        matches = re.findall(test_pattern, stdout)
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_name, class_name, status in matches:
            test_result = TestResult(
                passed=(status == "ok"),
                test_name=f"{class_name}.{test_name}",
            )
            
            if status == "ok":
                passed += 1
            elif status == "FAIL":
                failed += 1
            else:
                skipped += 1
            
            results.append(test_result)
        
        total = passed + failed + skipped
        
        return TestRunResult(
            success=success,
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            output=stdout,
            error_output=stderr,
            framework="unittest",
        )
    
    def _parse_jest_output(
        self,
        stdout: str,
        stderr: str,
        success: bool,
    ) -> TestRunResult:
        """Parse Jest output."""
        results: List[TestResult] = []
        
        # Extract test results
        # Pattern: PASS/FAIL test_name
        test_pattern = r"(PASS|FAIL)\s+(.+)"
        matches = re.findall(test_pattern, stdout)
        
        passed = 0
        failed = 0
        
        for status, test_name in matches:
            test_result = TestResult(
                passed=(status == "PASS"),
                test_name=test_name.strip(),
            )
            
            if status == "PASS":
                passed += 1
            else:
                failed += 1
            
            results.append(test_result)
        
        total = passed + failed
        
        return TestRunResult(
            success=success,
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=0,
            results=results,
            output=stdout,
            error_output=stderr,
            framework="jest",
        )
    
    def run_test_for_function(
        self,
        function_name: str,
        source_file: Path,
    ) -> Optional[TestRunResult]:
        """
        Run tests for a specific function.
        
        Args:
            function_name: Name of function.
            source_file: Source file containing function.
        
        Returns:
            TestRunResult if tests found, None otherwise.
        """
        # Find test file
        from agent.test_intelligence import create_test_intelligence
        
        intelligence = create_test_intelligence(self.repo_root)
        test_file = intelligence.find_test_file(source_file)
        
        if not test_file:
            return None
        
        # Run tests matching function name
        return self.run_tests(test_path=test_file, test_name=function_name)


def create_test_runner(repo_root: Path, framework: Optional[str] = None) -> TestRunner:
    """
    Create test runner instance.
    
    Args:
        repo_root: Repository root directory.
        framework: Optional test framework.
    
    Returns:
        TestRunner instance.
    """
    return TestRunner(repo_root, framework)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    runner = create_test_runner(repo_root)
    
    result = runner.run_tests(verbose=True)
    print(f"Tests: {result.total_tests}, Passed: {result.passed}, Failed: {result.failed}")
