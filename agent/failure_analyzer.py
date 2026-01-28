"""Analyze test failures and provide actionable insights."""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from agent.test_runner import TestFailure, TestResult


class FailureCategory(Enum):
    """Categories of test failures."""
    ASSERTION_ERROR = "assertion_error"
    ATTRIBUTE_ERROR = "attribute_error"
    TYPE_ERROR = "type_error"
    NAME_ERROR = "name_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    VALUE_ERROR = "value_error"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    INDENTATION_ERROR = "indentation_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Analysis of a test failure."""
    
    failure: TestFailure
    category: FailureCategory
    root_cause: str
    suggested_fix: str
    confidence: float  # 0.0 to 1.0
    relevant_lines: List[str] = None
    
    def __post_init__(self):
        if self.relevant_lines is None:
            self.relevant_lines = []


class FailureAnalyzer:
    """Analyze test failures and suggest fixes."""
    
    def __init__(self):
        """Initialize failure analyzer."""
        self.category_patterns = {
            FailureCategory.ASSERTION_ERROR: [
                r'AssertionError',
                r'assert\s+.+\s+==\s+.+',
                r'Expected .+ but got .+',
            ],
            FailureCategory.ATTRIBUTE_ERROR: [
                r'AttributeError',
                r"has no attribute",
                r"object has no attribute",
            ],
            FailureCategory.TYPE_ERROR: [
                r'TypeError',
                r"argument .+ must be",
                r"takes .+ positional argument",
                r"unsupported operand type",
            ],
            FailureCategory.NAME_ERROR: [
                r'NameError',
                r"name '.*' is not defined",
            ],
            FailureCategory.KEY_ERROR: [
                r'KeyError',
                r"key '.*' not found",
            ],
            FailureCategory.INDEX_ERROR: [
                r'IndexError',
                r"list index out of range",
                r"index out of bounds",
            ],
            FailureCategory.VALUE_ERROR: [
                r'ValueError',
                r"invalid literal",
                r"could not convert",
            ],
            FailureCategory.IMPORT_ERROR: [
                r'ImportError',
                r'ModuleNotFoundError',
                r"No module named",
                r"cannot import name",
            ],
            FailureCategory.SYNTAX_ERROR: [
                r'SyntaxError',
                r"invalid syntax",
            ],
            FailureCategory.INDENTATION_ERROR: [
                r'IndentationError',
                r"unexpected indent",
                r"expected an indented block",
            ],
            FailureCategory.TIMEOUT: [
                r'TimeoutError',
                r'timeout',
                r'timed out',
            ],
        }
    
    def analyze_failure(self, failure: TestFailure) -> FailureAnalysis:
        """
        Analyze a single test failure.
        
        Args:
            failure: TestFailure to analyze.
        
        Returns:
            FailureAnalysis with insights.
        """
        # Categorize the failure
        category = self._categorize_failure(failure)
        
        # Determine root cause
        root_cause = self._determine_root_cause(failure, category)
        
        # Suggest fix
        suggested_fix = self._suggest_fix(failure, category)
        
        # Calculate confidence
        confidence = self._calculate_confidence(failure, category)
        
        # Extract relevant lines
        relevant_lines = self._extract_relevant_lines(failure)
        
        return FailureAnalysis(
            failure=failure,
            category=category,
            root_cause=root_cause,
            suggested_fix=suggested_fix,
            confidence=confidence,
            relevant_lines=relevant_lines
        )
    
    def analyze_result(self, result: TestResult) -> List[FailureAnalysis]:
        """
        Analyze all failures in a test result.
        
        Args:
            result: TestResult to analyze.
        
        Returns:
            List of FailureAnalysis.
        """
        analyses = []
        for failure in result.failures:
            analysis = self.analyze_failure(failure)
            analyses.append(analysis)
        return analyses
    
    def _categorize_failure(self, failure: TestFailure) -> FailureCategory:
        """Categorize the failure based on error patterns."""
        error_text = f"{failure.error_type} {failure.error_message} {failure.traceback}"
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_text, re.IGNORECASE):
                    return category
        
        return FailureCategory.UNKNOWN
    
    def _determine_root_cause(self, failure: TestFailure, category: FailureCategory) -> str:
        """Determine the root cause of the failure."""
        if category == FailureCategory.ASSERTION_ERROR:
            return self._analyze_assertion_error(failure)
        elif category == FailureCategory.ATTRIBUTE_ERROR:
            return self._analyze_attribute_error(failure)
        elif category == FailureCategory.TYPE_ERROR:
            return self._analyze_type_error(failure)
        elif category == FailureCategory.NAME_ERROR:
            return self._analyze_name_error(failure)
        elif category == FailureCategory.KEY_ERROR:
            return self._analyze_key_error(failure)
        elif category == FailureCategory.INDEX_ERROR:
            return self._analyze_index_error(failure)
        elif category == FailureCategory.IMPORT_ERROR:
            return self._analyze_import_error(failure)
        elif category == FailureCategory.SYNTAX_ERROR:
            return self._analyze_syntax_error(failure)
        else:
            return f"Test failed with {failure.error_type}: {failure.error_message}"
    
    def _suggest_fix(self, failure: TestFailure, category: FailureCategory) -> str:
        """Suggest a fix for the failure."""
        if category == FailureCategory.ASSERTION_ERROR:
            return "Review the assertion logic and expected values. Check if the implementation matches the test expectations."
        
        elif category == FailureCategory.ATTRIBUTE_ERROR:
            # Try to extract the missing attribute
            attr_match = re.search(r"has no attribute '(\w+)'", failure.error_message)
            if attr_match:
                attr_name = attr_match.group(1)
                return f"Add the missing attribute '{attr_name}' to the class or check for typos in the attribute name."
            return "Check if the object is correctly initialized and has the expected attributes."
        
        elif category == FailureCategory.TYPE_ERROR:
            # Check if it's about wrong number of arguments
            if "positional argument" in failure.error_message:
                return "Fix the function signature or update the function call to match the expected number of arguments."
            return "Ensure the correct types are being passed to the function. Add type checking or conversion if needed."
        
        elif category == FailureCategory.NAME_ERROR:
            # Extract the undefined name
            name_match = re.search(r"name '(\w+)' is not defined", failure.error_message)
            if name_match:
                name = name_match.group(1)
                return f"Define the variable '{name}' before using it, or check for typos. Consider importing it if it's from another module."
            return "Check for undefined variables or missing imports."
        
        elif category == FailureCategory.KEY_ERROR:
            # Extract the missing key
            key_match = re.search(r"'(\w+)'", failure.error_message)
            if key_match:
                key = key_match.group(1)
                return f"Ensure the key '{key}' exists in the dictionary. Use .get() method with a default value or check with 'in' operator."
            return "Check if the dictionary key exists before accessing it. Use dict.get() or verify keys."
        
        elif category == FailureCategory.INDEX_ERROR:
            return "Check list bounds before accessing. Ensure the list is not empty and the index is within range."
        
        elif category == FailureCategory.IMPORT_ERROR:
            # Extract the module name
            module_match = re.search(r"No module named '([\w.]+)'", failure.error_message)
            if module_match:
                module = module_match.group(1)
                return f"Install the missing module '{module}' using pip or check if the module path is correct."
            return "Check if the module is installed and the import path is correct."
        
        elif category == FailureCategory.SYNTAX_ERROR:
            return "Fix the syntax error at the indicated line. Check for missing colons, parentheses, or quotes."
        
        elif category == FailureCategory.INDENTATION_ERROR:
            return "Fix the indentation. Ensure consistent use of spaces (4 spaces recommended) and proper block structure."
        
        else:
            return "Review the error message and traceback to identify the issue. Check the implementation logic."
    
    def _calculate_confidence(self, failure: TestFailure, category: FailureCategory) -> float:
        """Calculate confidence in the analysis."""
        if category == FailureCategory.UNKNOWN:
            return 0.3
        
        # Higher confidence for well-structured error messages
        if failure.file_path and failure.line_number:
            confidence = 0.8
        elif failure.error_message:
            confidence = 0.6
        else:
            confidence = 0.4
        
        # Boost confidence for specific error types
        if category in [
            FailureCategory.SYNTAX_ERROR,
            FailureCategory.INDENTATION_ERROR,
            FailureCategory.IMPORT_ERROR,
            FailureCategory.NAME_ERROR,
        ]:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_relevant_lines(self, failure: TestFailure) -> List[str]:
        """Extract relevant lines from the traceback."""
        if not failure.traceback:
            return []
        
        lines = []
        for line in failure.traceback.split("\n"):
            # Extract lines that look like code or error messages
            if any(keyword in line for keyword in ["File", "line", "assert", "Error", "    "]):
                lines.append(line.strip())
        
        return lines[:10]  # Limit to 10 most relevant lines
    
    def _analyze_assertion_error(self, failure: TestFailure) -> str:
        """Analyze assertion errors in detail."""
        msg = failure.error_message
        
        # Check for common assertion patterns
        if "!=" in msg or "==" in msg:
            # Value comparison
            return "Value mismatch: The actual value doesn't match the expected value."
        elif "is not" in msg or "is None" in msg:
            return "Identity/None check failed: Object is not what was expected."
        elif "not in" in msg or "in" in msg:
            return "Membership test failed: Item is/isn't in the collection as expected."
        elif failure.assertion_details:
            return f"Assertion failed: {failure.assertion_details}"
        else:
            return "Assertion failed: The test condition was not met."
    
    def _analyze_attribute_error(self, failure: TestFailure) -> str:
        """Analyze attribute errors."""
        if "NoneType" in failure.error_message:
            return "Attempting to access an attribute on None. The object was not properly initialized or returned None."
        return f"Missing or inaccessible attribute: {failure.error_message}"
    
    def _analyze_type_error(self, failure: TestFailure) -> str:
        """Analyze type errors."""
        msg = failure.error_message
        
        if "takes" in msg and "positional argument" in msg:
            return "Function signature mismatch: Wrong number of arguments provided."
        elif "unsupported operand" in msg:
            return "Type incompatibility: Trying to perform an operation on incompatible types."
        else:
            return f"Type error: {msg}"
    
    def _analyze_name_error(self, failure: TestFailure) -> str:
        """Analyze name errors."""
        name_match = re.search(r"name '(\w+)' is not defined", failure.error_message)
        if name_match:
            name = name_match.group(1)
            return f"Variable '{name}' is used before definition or is misspelled."
        return "Variable or name is not defined in the current scope."
    
    def _analyze_key_error(self, failure: TestFailure) -> str:
        """Analyze key errors."""
        return f"Dictionary key not found: {failure.error_message}"
    
    def _analyze_index_error(self, failure: TestFailure) -> str:
        """Analyze index errors."""
        return "List index out of range: Attempting to access an index that doesn't exist."
    
    def _analyze_import_error(self, failure: TestFailure) -> str:
        """Analyze import errors."""
        module_match = re.search(r"No module named '([\w.]+)'", failure.error_message)
        if module_match:
            module = module_match.group(1)
            return f"Module '{module}' is not installed or not found in the Python path."
        return f"Import failed: {failure.error_message}"
    
    def _analyze_syntax_error(self, failure: TestFailure) -> str:
        """Analyze syntax errors."""
        if failure.line_number:
            return f"Syntax error at line {failure.line_number}: Invalid Python syntax."
        return "Syntax error: The code contains invalid Python syntax."
    
    def generate_fix_prompt(self, analysis: FailureAnalysis) -> str:
        """
        Generate a prompt for the LLM to fix the failure.
        
        Args:
            analysis: FailureAnalysis to generate prompt from.
        
        Returns:
            Prompt string for the LLM.
        """
        failure = analysis.failure
        
        prompt_parts = [
            "# Test Failure Analysis",
            "",
            f"**Test**: {failure.test_name}",
            f"**Error Type**: {failure.error_type}",
            f"**Error Message**: {failure.error_message}",
            "",
            f"**Root Cause**: {analysis.root_cause}",
            f"**Category**: {analysis.category.value}",
            "",
            f"**Suggested Fix**: {analysis.suggested_fix}",
            "",
        ]
        
        if failure.file_path:
            prompt_parts.append(f"**File**: {failure.file_path}")
            if failure.line_number:
                prompt_parts.append(f"**Line**: {failure.line_number}")
            prompt_parts.append("")
        
        if analysis.relevant_lines:
            prompt_parts.append("**Relevant Code**:")
            prompt_parts.append("```")
            prompt_parts.extend(analysis.relevant_lines)
            prompt_parts.append("```")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Please generate a fix for this test failure. Focus on:",
            "1. Addressing the root cause identified above",
            "2. Ensuring the fix is minimal and targeted",
            "3. Preserving existing functionality",
            "4. Following the project's coding style",
            "",
            "Provide the fix as a unified diff."
        ])
        
        return "\n".join(prompt_parts)
    
    def summarize_failures(self, analyses: List[FailureAnalysis]) -> Dict[str, Any]:
        """
        Summarize multiple failure analyses.
        
        Args:
            analyses: List of FailureAnalysis.
        
        Returns:
            Dictionary with summary statistics.
        """
        if not analyses:
            return {
                "total_failures": 0,
                "categories": {},
                "avg_confidence": 0.0,
                "common_issues": []
            }
        
        # Count by category
        categories = {}
        for analysis in analyses:
            cat = analysis.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(a.confidence for a in analyses) / len(analyses)
        
        # Identify common issues (categories with multiple occurrences)
        common_issues = [cat for cat, count in categories.items() if count > 1]
        
        return {
            "total_failures": len(analyses),
            "categories": categories,
            "avg_confidence": avg_confidence,
            "common_issues": common_issues,
            "high_confidence_fixes": [
                a for a in analyses if a.confidence > 0.7
            ]
        }
