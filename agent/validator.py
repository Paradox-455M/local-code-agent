"""Pre-apply validation and self-correction."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from tools.apply_patch import PatchError
    from core.exceptions import ExecutionError
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from tools.apply_patch import PatchError
    from core.exceptions import ExecutionError


@dataclass
class ValidationResult:
    """Result of diff validation."""

    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    corrected_diff: Optional[str] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []


def validate_diff_syntax(diff_text: str, file_path: str, repo_root: str) -> ValidationResult:
    """
    Validate that a diff will produce valid Python syntax.

    Args:
        diff_text: The unified diff text.
        file_path: Path to the file being modified.
        repo_root: Repository root directory.

    Returns:
        ValidationResult with validation status and any errors.
    """
    result = ValidationResult(valid=True)
    
    if not file_path.endswith(".py"):
        # Non-Python files: basic validation only
        return result
    
    # Try to parse the diff and check syntax
    try:
        # Extract the new content from diff
        new_content = _extract_new_content_from_diff(diff_text)
        if new_content:
            # Try to parse as Python AST
            try:
                ast.parse(new_content)
            except SyntaxError as e:
                result.valid = False
                result.errors.append(f"Syntax error in generated code: {e}")
                result.suggestions.append(f"Fix syntax error at line {e.lineno}: {e.msg}")
    except Exception as e:
        result.warnings.append(f"Could not validate syntax: {e}")
    
    return result


def validate_diff_structure(diff_text: str, file_path: str, repo_root: str) -> ValidationResult:
    """
    Validate that a diff matches the file structure.

    Args:
        diff_text: The unified diff text.
        file_path: Path to the file being modified.
        repo_root: Repository root directory.

    Returns:
        ValidationResult with validation status.
    """
    result = ValidationResult(valid=True)
    full_path = Path(repo_root) / file_path
    
    if not full_path.exists():
        # New file: no structure validation needed
        return result
    
    try:
        original_content = full_path.read_text(encoding="utf-8")
        
        # Check if diff references existing lines correctly
        # This is a basic check - full validation happens in apply_patch
        lines = original_content.splitlines()
        
        # Extract hunk information
        import re
        hunk_pattern = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        
        for line in diff_text.splitlines():
            if line.startswith("@@"):
                match = hunk_pattern.match(line)
                if match:
                    old_start = int(match.group(1))
                    old_len = int(match.group(2) or "1")
                    
                    # Check if referenced lines exist
                    if old_start > len(lines):
                        result.warnings.append(
                            f"Hunk references line {old_start} but file only has {len(lines)} lines"
                        )
    except Exception as e:
        result.warnings.append(f"Could not validate structure: {e}")
    
    return result


def auto_correct_diff(diff_text: str, file_path: str, repo_root: str) -> Tuple[str, List[str]]:
    """
    Attempt to auto-correct common mistakes in diffs.

    Args:
        diff_text: The unified diff text.
        file_path: Path to the file being modified.
        repo_root: Repository root directory.

    Returns:
        Tuple of (corrected_diff, corrections_applied).
    """
    corrected = diff_text
    corrections = []
    
    # Common fixes:
    # 1. Fix trailing whitespace in context lines
    lines = corrected.splitlines()
    fixed_lines = []
    for i, line in enumerate(lines):
        if line.startswith(" ") and line.endswith(" ") and len(line) > 1:
            # Context line with trailing space - remove it
            fixed_lines.append(line.rstrip())
            if i == 0 or lines[i-1] != line.rstrip():
                corrections.append("Removed trailing whitespace")
        else:
            fixed_lines.append(line)
    
    corrected = "\n".join(fixed_lines)
    
    # 2. Ensure proper line endings
    if not corrected.endswith("\n"):
        corrected += "\n"
        corrections.append("Added missing newline")
    
    return corrected, corrections


def validate_before_apply(
    diff_text: str, file_path: str, repo_root: str, auto_correct: bool = True
) -> ValidationResult:
    """
    Comprehensive validation before applying a diff.

    Args:
        diff_text: The unified diff text.
        file_path: Path to the file being modified.
        repo_root: Repository root directory.
        auto_correct: Whether to attempt auto-correction.

    Returns:
        ValidationResult with validation status and corrections.
    """
    # Run all validations
    syntax_result = validate_diff_syntax(diff_text, file_path, repo_root)
    structure_result = validate_diff_structure(diff_text, file_path, repo_root)
    
    # Combine results
    result = ValidationResult(valid=True)
    result.errors.extend(syntax_result.errors)
    result.errors.extend(structure_result.errors)
    result.warnings.extend(syntax_result.warnings)
    result.warnings.extend(structure_result.warnings)
    
    # If there are errors, try auto-correction
    if result.errors and auto_correct:
        corrected, corrections = auto_correct_diff(diff_text, file_path, repo_root)
        result.corrected_diff = corrected
        if corrections:
            result.suggestions.append(f"Auto-corrections applied: {', '.join(corrections)}")
            # Re-validate corrected version
            corrected_syntax = validate_diff_syntax(corrected, file_path, repo_root)
            if not corrected_syntax.errors:
                result.valid = True
                result.errors = []
    
    result.valid = len(result.errors) == 0
    
    return result


def _extract_new_content_from_diff(diff_text: str) -> Optional[str]:
    """Extract the new file content from a unified diff."""
    lines = diff_text.splitlines()
    new_lines = []
    in_diff = False
    
    for line in lines:
        if line.startswith("+++ "):
            in_diff = True
            continue
        if line.startswith("--- ") and not line.startswith("+++ "):
            continue
        if line.startswith("@@"):
            continue
        
        if in_diff:
            if line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith(" "):
                new_lines.append(line[1:])
            elif line.startswith("-"):
                # Skip deleted lines
                continue
    
    return "\n".join(new_lines) if new_lines else None


def run_syntax_check(file_path: str, repo_root: str) -> Tuple[bool, str]:
    """
    Run Python syntax check on a file.

    Args:
        file_path: Path to the file.
        repo_root: Repository root directory.

    Returns:
        Tuple of (is_valid, error_message).
    """
    full_path = Path(repo_root) / file_path
    
    if not full_path.exists() or not file_path.endswith(".py"):
        return True, ""
    
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", str(full_path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    # Demo
    test_diff = """--- test.py
+++ test.py
@@ -1 +1 @@
-def hello():
+def hello():  # Added comment
     pass
"""
    result = validate_before_apply(test_diff, "test.py", ".")
    print(f"Valid: {result.valid}")
    print(f"Errors: {result.errors}")
