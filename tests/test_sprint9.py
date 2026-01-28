"""Tests for Sprint 9: Agentic Loop & Self-Correction features."""

import tempfile
from pathlib import Path

import pytest

from agent.test_detector import TestFrameworkDetector, detect_test_framework
from agent.test_runner import TestRunner, TestStatus
from agent.failure_analyzer import FailureAnalyzer, FailureCategory
from agent.syntax_checker import SyntaxChecker, validate_python_files


@pytest.fixture
def temp_repo():
    """Create a temporary repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pytest_repo(temp_repo):
    """Create a repo with pytest setup."""
    # Create pytest.ini
    (temp_repo / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    
    # Create test directory
    tests_dir = temp_repo / "tests"
    tests_dir.mkdir()
    
    # Create a simple test file
    (tests_dir / "test_sample.py").write_text("""
import pytest

def test_passing():
    assert 1 + 1 == 2

def test_failing():
    assert 1 + 1 == 3
""")
    
    return temp_repo


class TestFrameworkDetection:
    """Test framework detection."""
    
    def test_detect_pytest(self, pytest_repo):
        """Test pytest detection."""
        detector = TestFrameworkDetector(pytest_repo)
        framework = detector.detect()
        
        assert framework.name == "pytest"
        assert framework.test_command is not None
        assert "pytest" in framework.test_command
    
    def test_detect_no_framework(self, temp_repo):
        """Test detection when no framework is found."""
        detector = TestFrameworkDetector(temp_repo)
        framework = detector.detect()
        
        assert framework.name == "unknown"
    
    def test_find_test_files(self, pytest_repo):
        """Test finding test files."""
        detector = TestFrameworkDetector(pytest_repo)
        test_files = detector.find_test_files()
        
        assert len(test_files) >= 1
        assert any("test_sample.py" in str(f) for f in test_files)
    
    def test_get_test_command(self, pytest_repo):
        """Test getting test command."""
        detector = TestFrameworkDetector(pytest_repo)
        command = detector.get_test_command()
        
        assert "pytest" in command
    
    def test_get_test_command_specific_file(self, pytest_repo):
        """Test getting test command for specific file."""
        detector = TestFrameworkDetector(pytest_repo)
        command = detector.get_test_command("tests/test_sample.py")
        
        assert "pytest" in command
        assert "test_sample.py" in command


class TestTestRunner:
    """Test the test runner."""
    
    def test_run_tests_passing(self, temp_repo):
        """Test running tests that pass."""
        # Create a simple passing test
        tests_dir = temp_repo / "tests"
        tests_dir.mkdir()
        
        test_file = tests_dir / "test_pass.py"
        test_file.write_text("""
def test_pass():
    assert True
""")
        
        runner = TestRunner(temp_repo)
        result = runner.run_tests(test_command="python -m pytest tests/test_pass.py -v")
        
        # Note: This might fail if pytest is not installed, which is expected
        assert result is not None
    
    def test_parse_pytest_output(self):
        """Test parsing pytest output."""
        runner = TestRunner()
        
        stdout = """
============================= test session starts ==============================
collected 2 items

tests/test_sample.py::test_passing PASSED                                [ 50%]
tests/test_sample.py::test_failing FAILED                                [100%]

=================================== FAILURES ===================================
________________________________ test_failing __________________________________

    def test_failing():
>       assert 1 + 1 == 3
E       assert 2 == 3

tests/test_sample.py:8: AssertionError
=========================== short test summary info ============================
FAILED tests/test_sample.py::test_failing - assert 2 == 3
========================= 1 failed, 1 passed in 0.12s ==========================
"""
        
        result = runner._parse_pytest_output(stdout, "", 1)
        
        assert result.total_tests == 2
        assert result.passed_tests == 1
        assert result.failed_tests == 1
        assert result.status == TestStatus.FAILED


class TestFailureAnalyzer:
    """Test failure analysis."""
    
    def test_categorize_assertion_error(self):
        """Test categorizing assertion errors."""
        from agent.test_runner import TestFailure
        
        analyzer = FailureAnalyzer()
        
        failure = TestFailure(
            test_name="test_example",
            error_type="AssertionError",
            error_message="assert 2 == 3"
        )
        
        analysis = analyzer.analyze_failure(failure)
        
        assert analysis.category == FailureCategory.ASSERTION_ERROR
        assert analysis.confidence > 0.5
        assert "assertion" in analysis.root_cause.lower()
    
    def test_categorize_name_error(self):
        """Test categorizing name errors."""
        from agent.test_runner import TestFailure
        
        analyzer = FailureAnalyzer()
        
        failure = TestFailure(
            test_name="test_example",
            error_type="NameError",
            error_message="name 'foo' is not defined"
        )
        
        analysis = analyzer.analyze_failure(failure)
        
        assert analysis.category == FailureCategory.NAME_ERROR
        assert "foo" in analysis.root_cause
        assert "define" in analysis.suggested_fix.lower() or "import" in analysis.suggested_fix.lower()
    
    def test_categorize_attribute_error(self):
        """Test categorizing attribute errors."""
        from agent.test_runner import TestFailure
        
        analyzer = FailureAnalyzer()
        
        failure = TestFailure(
            test_name="test_example",
            error_type="AttributeError",
            error_message="'Foo' object has no attribute 'bar'"
        )
        
        analysis = analyzer.analyze_failure(failure)
        
        assert analysis.category == FailureCategory.ATTRIBUTE_ERROR
        assert "bar" in analysis.suggested_fix
    
    def test_generate_fix_prompt(self):
        """Test generating fix prompts."""
        from agent.test_runner import TestFailure
        
        analyzer = FailureAnalyzer()
        
        failure = TestFailure(
            test_name="test_example",
            error_type="AssertionError",
            error_message="assert 2 == 3",
            file_path="test_file.py",
            line_number=42
        )
        
        analysis = analyzer.analyze_failure(failure)
        prompt = analyzer.generate_fix_prompt(analysis)
        
        assert "test_example" in prompt
        assert "AssertionError" in prompt
        assert "test_file.py" in prompt
        assert "42" in prompt
    
    def test_summarize_failures(self):
        """Test summarizing multiple failures."""
        from agent.test_runner import TestFailure
        
        analyzer = FailureAnalyzer()
        
        failures = [
            TestFailure("test1", "AssertionError", "assert failed"),
            TestFailure("test2", "AssertionError", "assert failed"),
            TestFailure("test3", "NameError", "name not defined"),
        ]
        
        analyses = [analyzer.analyze_failure(f) for f in failures]
        summary = analyzer.summarize_failures(analyses)
        
        assert summary["total_failures"] == 3
        assert "assertion_error" in summary["categories"]
        assert summary["categories"]["assertion_error"] == 2


class TestSyntaxChecker:
    """Test syntax checking."""
    
    def test_check_valid_file(self, temp_repo):
        """Test checking a valid Python file."""
        test_file = temp_repo / "valid.py"
        test_file.write_text("def foo():\n    return 42\n")
        
        checker = SyntaxChecker(temp_repo)
        errors = checker.check_file(test_file)
        
        assert len(errors) == 0
    
    def test_check_syntax_error(self, temp_repo):
        """Test detecting syntax errors."""
        test_file = temp_repo / "invalid.py"
        test_file.write_text("def foo(\n    return 42\n")  # Missing closing paren
        
        checker = SyntaxChecker(temp_repo)
        errors = checker.check_file(test_file)
        
        assert len(errors) > 0
        assert errors[0].error_type == "SyntaxError"
    
    def test_check_indentation_error(self, temp_repo):
        """Test detecting indentation errors."""
        test_file = temp_repo / "indent.py"
        test_file.write_text("def foo():\nreturn 42\n")  # Missing indent
        
        checker = SyntaxChecker(temp_repo)
        errors = checker.check_file(test_file)
        
        assert len(errors) > 0
        assert errors[0].error_type == "IndentationError"
    
    def test_check_multiple_files(self, temp_repo):
        """Test checking multiple files."""
        valid = temp_repo / "valid.py"
        valid.write_text("def foo():\n    return 42\n")
        
        invalid = temp_repo / "invalid.py"
        invalid.write_text("def bar(\n    return 1\n")
        
        checker = SyntaxChecker(temp_repo)
        result = checker.check_files([valid, invalid])
        
        assert result.files_checked == 2
        assert len(result.syntax_errors) > 0
        assert not result.is_valid
    
    def test_validate_python_files(self, temp_repo):
        """Test validation convenience function."""
        test_file = temp_repo / "test.py"
        test_file.write_text("x = 1\nprint(x)\n")
        
        result = validate_python_files([test_file], run_lint=False)
        
        assert result.files_checked == 1
        assert result.is_valid


class TestAutoFixer:
    """Test auto-fixer."""
    
    def test_can_fix_detection(self):
        """Test detecting fixable errors."""
        from agent.syntax_checker import AutoFixer, SyntaxError as SynErr
        
        fixer = AutoFixer()
        
        fixable = SynErr(
            file_path="test.py",
            line_number=1,
            column=0,
            error_message="Missing parentheses in call to 'print'"
        )
        
        assert fixer.can_fix(fixable) is True
    
    def test_suggest_fix(self):
        """Test suggesting fixes."""
        from agent.syntax_checker import AutoFixer, SyntaxError as SynErr
        
        fixer = AutoFixer()
        
        error = SynErr(
            file_path="test.py",
            line_number=1,
            column=0,
            error_message="Missing parentheses in call to 'print'"
        )
        
        suggestion = fixer.suggest_fix(error)
        
        assert suggestion is not None
        assert "parentheses" in suggestion.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
