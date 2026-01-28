"""Test intelligence - Claude Code level test understanding and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import ast
import re


@dataclass
class TestFramework:
    """Detected test framework information."""
    
    name: str  # 'pytest', 'unittest', 'jest', 'mocha', etc.
    version: Optional[str] = None
    patterns: List[str] = field(default_factory=list)  # Common patterns used
    fixtures: List[str] = field(default_factory=list)  # Common fixtures


@dataclass
class TestPattern:
    """Detected test pattern."""
    
    pattern_type: str  # 'fixture', 'parametrize', 'mock', 'async', etc.
    usage_count: int = 0
    examples: List[str] = field(default_factory=list)


@dataclass
class TestStructure:
    """Structure of test files."""
    
    test_class_prefix: str = "Test"  # e.g., "Test" for unittest
    test_function_prefix: str = "test_"  # e.g., "test_" for pytest
    uses_classes: bool = False
    uses_fixtures: bool = False
    uses_parametrize: bool = False
    uses_mocks: bool = False
    async_tests: bool = False


class TestIntelligence:
    """Deep understanding of test structure and patterns."""
    
    def __init__(self, repo_root: Path):
        """
        Initialize test intelligence.
        
        Args:
            repo_root: Repository root directory.
        """
        self.repo_root = Path(repo_root).resolve()
        self.framework: Optional[TestFramework] = None
        self.structure: Optional[TestStructure] = None
        self.patterns: Dict[str, TestPattern] = {}
        self._analyze_tests()
    
    def _analyze_tests(self) -> None:
        """Analyze test files to understand patterns."""
        test_files = self._find_test_files()
        
        if not test_files:
            # Default to pytest for Python
            self.framework = TestFramework(name="pytest")
            self.structure = TestStructure()
            return
        
        # Detect framework
        self.framework = self._detect_framework(test_files)
        
        # Analyze structure
        self.structure = self._analyze_structure(test_files)
        
        # Detect patterns
        self.patterns = self._detect_patterns(test_files)
    
    def _find_test_files(self) -> List[Path]:
        """Find all test files in repository."""
        test_files: List[Path] = []
        
        # Common test file patterns
        test_patterns = [
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/**/*.py",
            "**/__tests__/**/*.py",
            "**/*.test.js",
            "**/*.spec.js",
        ]
        
        for pattern in test_patterns:
            test_files.extend(self.repo_root.glob(pattern))
        
        # Also check for test directories
        test_dirs = ["tests", "test", "__tests__"]
        for test_dir in test_dirs:
            test_path = self.repo_root / test_dir
            if test_path.exists() and test_path.is_dir():
                test_files.extend(test_path.rglob("*.py"))
                test_files.extend(test_path.rglob("*.js"))
        
        return list(set(test_files))  # Remove duplicates
    
    def _detect_framework(self, test_files: List[Path]) -> TestFramework:
        """Detect test framework from test files."""
        framework_hints: Dict[str, int] = {}
        
        for test_file in test_files[:20]:  # Analyze first 20 files
            try:
                content = test_file.read_text(encoding="utf-8", errors="replace")
                
                # Check imports
                if "import pytest" in content or "from pytest" in content:
                    framework_hints["pytest"] = framework_hints.get("pytest", 0) + 1
                if "import unittest" in content or "from unittest" in content:
                    framework_hints["unittest"] = framework_hints.get("unittest", 0) + 1
                if "import jest" in content or "from jest" in content or "describe(" in content:
                    framework_hints["jest"] = framework_hints.get("jest", 0) + 1
                if "import mocha" in content or "from mocha" in content:
                    framework_hints["mocha"] = framework_hints.get("mocha", 0) + 1
                
                # Check for pytest markers
                if "@pytest.mark" in content or "pytest.fixture" in content:
                    framework_hints["pytest"] = framework_hints.get("pytest", 0) + 2
                
            except Exception:
                continue
        
        # Determine framework
        if framework_hints:
            detected = max(framework_hints.items(), key=lambda x: x[1])
            return TestFramework(name=detected[0])
        
        # Default based on file extension
        if any(f.suffix == ".js" for f in test_files):
            return TestFramework(name="jest")
        
        return TestFramework(name="pytest")  # Default for Python
    
    def _analyze_structure(self, test_files: List[Path]) -> TestStructure:
        """Analyze test file structure."""
        structure = TestStructure()
        
        for test_file in test_files[:10]:  # Analyze first 10 files
            try:
                if test_file.suffix == ".py":
                    structure = self._analyze_python_structure(test_file, structure)
                elif test_file.suffix == ".js":
                    structure = self._analyze_js_structure(test_file, structure)
            except Exception:
                continue
        
        return structure
    
    def _analyze_python_structure(self, test_file: Path, structure: TestStructure) -> TestStructure:
        """Analyze Python test file structure."""
        try:
            content = test_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for test classes
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("Test"):
                        structure.uses_classes = True
                        structure.test_class_prefix = "Test"
                
                # Check for test functions
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_"):
                        structure.test_function_prefix = "test_"
                    if "async" in [d.id for d in node.decorator_list if isinstance(d, ast.Name)]:
                        structure.async_tests = True
                
                # Check for fixtures
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Attribute):
                                if decorator.func.attr == "fixture":
                                    structure.uses_fixtures = True
                
                # Check for parametrize
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Attribute):
                                if decorator.func.attr == "parametrize":
                                    structure.uses_parametrize = True
                
                # Check for mocks
                if isinstance(node, ast.ImportFrom):
                    if node.module and "mock" in node.module.lower():
                        structure.uses_mocks = True
            
        except SyntaxError:
            pass
        
        return structure
    
    def _analyze_js_structure(self, test_file: Path, structure: TestStructure) -> TestStructure:
        """Analyze JavaScript test file structure."""
        try:
            content = test_file.read_text(encoding="utf-8", errors="replace")
            
            # Check for describe/it blocks (Jest/Mocha)
            if "describe(" in content:
                structure.uses_classes = True
            if "it(" in content or "test(" in content:
                structure.test_function_prefix = "test"
            if "beforeEach" in content or "afterEach" in content:
                structure.uses_fixtures = True
            if "jest.mock" in content or "sinon" in content:
                structure.uses_mocks = True
        
        except Exception:
            pass
        
        return structure
    
    def _detect_patterns(self, test_files: List[Path]) -> Dict[str, TestPattern]:
        """Detect common test patterns."""
        patterns: Dict[str, TestPattern] = {}
        
        for test_file in test_files[:10]:
            try:
                content = test_file.read_text(encoding="utf-8", errors="replace")
                
                # Detect fixture pattern
                if "fixture" in content.lower() or "@pytest.fixture" in content:
                    if "fixture" not in patterns:
                        patterns["fixture"] = TestPattern(pattern_type="fixture")
                    patterns["fixture"].usage_count += 1
                
                # Detect parametrize pattern
                if "parametrize" in content.lower() or "@pytest.mark.parametrize" in content:
                    if "parametrize" not in patterns:
                        patterns["parametrize"] = TestPattern(pattern_type="parametrize")
                    patterns["parametrize"].usage_count += 1
                
                # Detect mock pattern
                if "mock" in content.lower() or "Mock(" in content:
                    if "mock" not in patterns:
                        patterns["mock"] = TestPattern(pattern_type="mock")
                    patterns["mock"].usage_count += 1
                
            except Exception:
                continue
        
        return patterns
    
    def generate_test(
        self,
        function_name: str,
        function_code: str,
        test_name: Optional[str] = None,
    ) -> str:
        """
        Generate a test for a function following project conventions.
        
        Args:
            function_name: Name of function to test.
            function_code: Code of function to test.
            test_name: Optional custom test name.
        
        Returns:
            Generated test code.
        """
        if not self.framework or not self.structure:
            return self._generate_default_test(function_name, function_code)
        
        if self.framework.name == "pytest":
            return self._generate_pytest_test(function_name, function_code, test_name)
        elif self.framework.name == "unittest":
            return self._generate_unittest_test(function_name, function_code, test_name)
        elif self.framework.name == "jest":
            return self._generate_jest_test(function_name, function_code, test_name)
        else:
            return self._generate_default_test(function_name, function_code)
    
    def _generate_pytest_test(
        self,
        function_name: str,
        function_code: str,
        test_name: Optional[str],
    ) -> str:
        """Generate pytest test."""
        test_func_name = test_name or f"test_{function_name}"
        
        # Extract function signature
        try:
            tree = ast.parse(function_code)
            func_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    func_node = node
                    break
            
            if func_node:
                # Generate test based on function signature
                params = [arg.arg for arg in func_node.args.args if arg.arg != "self"]
                
                test_code = f"def {test_func_name}():\n"
                test_code += f'    """Test {function_name}."""\n'
                
                # Add basic test
                if params:
                    test_code += f"    # TODO: Add test cases\n"
                    test_code += f"    result = {function_name}({', '.join(['None'] * len(params))})\n"
                else:
                    test_code += f"    result = {function_name}()\n"
                
                test_code += "    assert result is not None\n"
                
                return test_code
        except Exception:
            pass
        
        # Fallback
        return f"def {test_func_name}():\n    assert {function_name}() is not None\n"
    
    def _generate_unittest_test(
        self,
        function_name: str,
        function_code: str,
        test_name: Optional[str],
    ) -> str:
        """Generate unittest test."""
        test_method_name = test_name or f"test_{function_name}"
        
        test_code = f"import unittest\n\n"
        test_code += f"class Test{function_name.capitalize()}(unittest.TestCase):\n"
        test_code += f'    """Test {function_name}."""\n\n'
        test_code += f"    def {test_method_name}(self):\n"
        test_code += f"        result = {function_name}()\n"
        test_code += f"        self.assertIsNotNone(result)\n"
        
        return test_code
    
    def _generate_jest_test(
        self,
        function_name: str,
        function_code: str,
        test_name: Optional[str],
    ) -> str:
        """Generate Jest test."""
        test_name_str = test_name or f"{function_name} works"
        
        test_code = f"describe('{function_name}', () => {{\n"
        test_code += f"  it('{test_name_str}', () => {{\n"
        test_code += f"    const result = {function_name}();\n"
        test_code += f"    expect(result).toBeDefined();\n"
        test_code += f"  }});\n"
        test_code += f"}});\n"
        
        return test_code
    
    def _generate_default_test(
        self,
        function_name: str,
        function_code: str,
    ) -> str:
        """Generate default test."""
        return f"def test_{function_name}():\n    assert {function_name}() is not None\n"
    
    def match_test_style(self, test_code: str) -> Tuple[bool, List[str]]:
        """
        Check if test code matches project style.
        
        Args:
            test_code: Test code to check.
        
        Returns:
            Tuple of (matches_style, violations).
        """
        violations: List[str] = []
        
        if not self.structure:
            return True, violations
        
        # Check naming convention
        if self.structure.test_function_prefix:
            if not any(f"def {self.structure.test_function_prefix}" in test_code for _ in [1]):
                violations.append(f"Test functions should start with '{self.structure.test_function_prefix}'")
        
        # Check class usage
        if self.structure.uses_classes and "class" not in test_code:
            violations.append("Project uses test classes, but test doesn't use a class")
        
        # Check fixture usage
        if self.structure.uses_fixtures and "fixture" not in test_code.lower():
            # Not a violation, just a suggestion
            pass
        
        return len(violations) == 0, violations
    
    def find_test_file(self, source_file: Path) -> Optional[Path]:
        """
        Find test file for a source file.
        
        Args:
            source_file: Source file path.
        
        Returns:
            Test file path if found.
        """
        # Common patterns
        source_name = source_file.stem
        source_dir = source_file.parent
        
        # Pattern 1: test_<name>.py in same directory
        test_file = source_dir / f"test_{source_name}.py"
        if test_file.exists():
            return test_file
        
        # Pattern 2: <name>_test.py in same directory
        test_file = source_dir / f"{source_name}_test.py"
        if test_file.exists():
            return test_file
        
        # Pattern 3: tests/test_<name>.py
        tests_dir = self.repo_root / "tests"
        if tests_dir.exists():
            test_file = tests_dir / f"test_{source_name}.py"
            if test_file.exists():
                return test_file
        
        # Pattern 4: tests/<dir>/test_<name>.py
        if tests_dir.exists():
            for subdir in tests_dir.iterdir():
                if subdir.is_dir():
                    test_file = subdir / f"test_{source_name}.py"
                    if test_file.exists():
                        return test_file
        
        return None


def create_test_intelligence(repo_root: Path) -> TestIntelligence:
    """
    Create test intelligence instance.
    
    Args:
        repo_root: Repository root directory.
    
    Returns:
        TestIntelligence instance.
    """
    return TestIntelligence(repo_root)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    intelligence = create_test_intelligence(repo_root)
    
    print(f"Framework: {intelligence.framework.name if intelligence.framework else 'unknown'}")
    print(f"Test prefix: {intelligence.structure.test_function_prefix if intelligence.structure else 'unknown'}")
    print(f"Uses classes: {intelligence.structure.uses_classes if intelligence.structure else False}")
