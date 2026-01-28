"""Syntax and lint checking for pre-flight validation."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class SyntaxError:
    """Information about a syntax error."""
    
    file_path: str
    line_number: int
    column: int
    error_message: str
    error_type: str = "SyntaxError"
    line_content: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [
            f"{self.file_path}:{self.line_number}:{self.column}",
            f"{self.error_type}: {self.error_message}"
        ]
        if self.line_content:
            parts.append(f"  {self.line_content.strip()}")
        return "\n".join(parts)


@dataclass
class LintIssue:
    """Information about a lint issue."""
    
    file_path: str
    line_number: int
    column: int
    code: str
    message: str
    severity: str  # error, warning, info
    
    def __str__(self) -> str:
        """String representation of the issue."""
        return f"{self.file_path}:{self.line_number}:{self.column}: {self.severity.upper()} [{self.code}] {self.message}"


@dataclass
class ValidationResult:
    """Results from syntax and lint validation."""
    
    syntax_errors: List[SyntaxError]
    lint_issues: List[LintIssue]
    files_checked: int = 0
    
    @property
    def has_syntax_errors(self) -> bool:
        """Check if there are syntax errors."""
        return len(self.syntax_errors) > 0
    
    @property
    def has_lint_errors(self) -> bool:
        """Check if there are lint errors (not warnings)."""
        return any(issue.severity == "error" for issue in self.lint_issues)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return not self.has_syntax_errors and not self.has_lint_errors
    
    def summary(self) -> str:
        """Get a summary of the validation."""
        if self.is_valid:
            return f"✅ All {self.files_checked} file(s) passed validation"
        else:
            parts = []
            if self.has_syntax_errors:
                parts.append(f"{len(self.syntax_errors)} syntax error(s)")
            if self.has_lint_errors:
                error_count = sum(1 for i in self.lint_issues if i.severity == "error")
                parts.append(f"{error_count} lint error(s)")
            return f"❌ Found: {', '.join(parts)}"


class SyntaxChecker:
    """Check Python syntax and run linting."""
    
    def __init__(self, repo_root: Path = None):
        """
        Initialize syntax checker.
        
        Args:
            repo_root: Root directory of the repository.
        """
        self.repo_root = repo_root or Path.cwd()
    
    def check_file(self, file_path: Path) -> List[SyntaxError]:
        """
        Check syntax of a single Python file.
        
        Args:
            file_path: Path to the file to check.
        
        Returns:
            List of SyntaxError objects.
        """
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Try to parse the file
            try:
                ast.parse(source, filename=str(file_path))
            except SyntaxError as e:
                # Extract line content
                line_content = None
                if e.lineno:
                    lines = source.split('\n')
                    if 0 <= e.lineno - 1 < len(lines):
                        line_content = lines[e.lineno - 1]
                
                errors.append(SyntaxError(
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    column=e.offset or 0,
                    error_message=e.msg or "Invalid syntax",
                    error_type="SyntaxError",
                    line_content=line_content
                ))
            except IndentationError as e:
                errors.append(SyntaxError(
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    column=e.offset or 0,
                    error_message=e.msg or "Invalid indentation",
                    error_type="IndentationError",
                ))
        
        except Exception as e:
            errors.append(SyntaxError(
                file_path=str(file_path),
                line_number=0,
                column=0,
                error_message=f"Failed to read file: {e}",
                error_type="FileError"
            ))
        
        return errors
    
    def check_files(self, file_paths: List[Path]) -> ValidationResult:
        """
        Check syntax of multiple files.
        
        Args:
            file_paths: List of file paths to check.
        
        Returns:
            ValidationResult with all errors.
        """
        all_errors = []
        
        for file_path in file_paths:
            if file_path.suffix == '.py':
                errors = self.check_file(file_path)
                all_errors.extend(errors)
        
        return ValidationResult(
            syntax_errors=all_errors,
            lint_issues=[],
            files_checked=len(file_paths)
        )
    
    def check_directory(self, directory: Path = None) -> ValidationResult:
        """
        Check syntax of all Python files in a directory.
        
        Args:
            directory: Directory to check. Defaults to repo root.
        
        Returns:
            ValidationResult with all errors.
        """
        directory = directory or self.repo_root
        
        # Find all Python files
        python_files = list(directory.rglob("*.py"))
        
        # Filter out common directories to skip
        skip_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', '.tox'}
        python_files = [
            f for f in python_files
            if not any(skip in f.parts for skip in skip_dirs)
        ]
        
        return self.check_files(python_files)
    
    def lint_with_ruff(self, file_paths: List[Path]) -> List[LintIssue]:
        """
        Run ruff linter on files.
        
        Args:
            file_paths: List of file paths to lint.
        
        Returns:
            List of LintIssue objects.
        """
        issues = []
        
        # Check if ruff is available
        try:
            result = subprocess.run(
                ['ruff', '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return issues
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return issues
        
        # Run ruff
        try:
            file_args = [str(f) for f in file_paths]
            result = subprocess.run(
                ['ruff', 'check', '--output-format=json'] + file_args,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.repo_root
            )
            
            # Parse JSON output
            import json
            if result.stdout:
                try:
                    ruff_results = json.loads(result.stdout)
                    for item in ruff_results:
                        issues.append(LintIssue(
                            file_path=item.get('filename', ''),
                            line_number=item.get('location', {}).get('row', 0),
                            column=item.get('location', {}).get('column', 0),
                            code=item.get('code', ''),
                            message=item.get('message', ''),
                            severity='error' if item.get('severity') == 'error' else 'warning'
                        ))
                except json.JSONDecodeError:
                    pass
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return issues
    
    def validate_files(
        self,
        file_paths: List[Path],
        run_lint: bool = True
    ) -> ValidationResult:
        """
        Full validation: syntax check + optional linting.
        
        Args:
            file_paths: List of file paths to validate.
            run_lint: Whether to run linting in addition to syntax check.
        
        Returns:
            ValidationResult with all issues.
        """
        # Check syntax
        syntax_result = self.check_files(file_paths)
        
        # Run linting if requested and no syntax errors
        lint_issues = []
        if run_lint and not syntax_result.has_syntax_errors:
            lint_issues = self.lint_with_ruff(file_paths)
        
        return ValidationResult(
            syntax_errors=syntax_result.syntax_errors,
            lint_issues=lint_issues,
            files_checked=len(file_paths)
        )
    
    def print_results(self, result: ValidationResult) -> None:
        """
        Print validation results in a formatted way.
        
        Args:
            result: ValidationResult to display.
        """
        console.print(f"\n[bold]Validation Results:[/bold] {result.summary()}\n")
        
        if result.syntax_errors:
            console.print("[bold red]Syntax Errors:[/bold red]")
            for error in result.syntax_errors:
                console.print(f"  [red]✗[/red] {error}")
            console.print()
        
        if result.lint_issues:
            # Separate errors and warnings
            errors = [i for i in result.lint_issues if i.severity == "error"]
            warnings = [i for i in result.lint_issues if i.severity == "warning"]
            
            if errors:
                console.print("[bold red]Lint Errors:[/bold red]")
                for issue in errors:
                    console.print(f"  [red]✗[/red] {issue}")
                console.print()
            
            if warnings:
                console.print("[bold yellow]Lint Warnings:[/bold yellow]")
                for issue in warnings[:10]:  # Limit warnings
                    console.print(f"  [yellow]⚠[/yellow] {issue}")
                if len(warnings) > 10:
                    console.print(f"  [dim]... and {len(warnings) - 10} more warnings[/dim]")
                console.print()


class AutoFixer:
    """Automatic fixer for common syntax and lint issues."""
    
    def __init__(self):
        """Initialize auto fixer."""
        pass
    
    def can_fix(self, error: SyntaxError) -> bool:
        """
        Check if an error can be automatically fixed.
        
        Args:
            error: SyntaxError to check.
        
        Returns:
            True if can be fixed automatically.
        """
        # Common fixable patterns
        fixable_patterns = [
            "missing parentheses in call",
            "invalid syntax. Perhaps you forgot a comma",
            "unterminated string",
            "unmatched ')'",
        ]
        
        return any(pattern in error.error_message.lower() for pattern in fixable_patterns)
    
    def suggest_fix(self, error: SyntaxError) -> Optional[str]:
        """
        Suggest a fix for a syntax error.
        
        Args:
            error: SyntaxError to fix.
        
        Returns:
            Suggested fix description or None.
        """
        msg = error.error_message.lower()
        
        if "missing parentheses" in msg:
            return "Add parentheses around the print statement"
        elif "forgot a comma" in msg:
            return "Add missing comma in list/tuple/dict"
        elif "unterminated string" in msg:
            return "Add closing quote for string"
        elif "unmatched" in msg:
            return "Fix unmatched parentheses/brackets"
        elif "expected an indented block" in msg:
            return "Add indentation after colon"
        elif "unexpected indent" in msg:
            return "Remove extra indentation"
        
        return None
    
    def fix_with_ruff(self, file_paths: List[Path]) -> bool:
        """
        Run ruff auto-fix on files.
        
        Args:
            file_paths: List of file paths to fix.
        
        Returns:
            True if fixes were applied.
        """
        try:
            file_args = [str(f) for f in file_paths]
            result = subprocess.run(
                ['ruff', 'check', '--fix'] + file_args,
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def validate_python_files(file_paths: List[Path], run_lint: bool = True) -> ValidationResult:
    """
    Convenience function to validate Python files.
    
    Args:
        file_paths: List of file paths to validate.
        run_lint: Whether to run linting.
    
    Returns:
        ValidationResult with all issues.
    """
    checker = SyntaxChecker()
    return checker.validate_files(file_paths, run_lint)
