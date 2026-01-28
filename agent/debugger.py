"""Debugger - Intelligent debugging assistance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import ast
import re


@dataclass
class DebugPoint:
    """Debug point location."""
    
    file_path: str
    line_number: int
    function_name: Optional[str] = None
    variable_names: List[str] = field(default_factory=list)


@dataclass
class DebugSuggestion:
    """Debugging suggestion."""
    
    location: DebugPoint
    suggestion_type: str  # 'log', 'breakpoint', 'inspect', 'test'
    code: str
    explanation: str


class Debugger:
    """Intelligent debugging assistance."""
    
    def __init__(self, repo_root: Path):
        """
        Initialize debugger.
        
        Args:
            repo_root: Repository root directory.
        """
        self.repo_root = Path(repo_root).resolve()
    
    def suggest_debug_points(
        self,
        error_location: str,
        error_type: str,
    ) -> List[DebugSuggestion]:
        """
        Suggest debug points for an error.
        
        Args:
            error_location: Error location (file:line).
            error_type: Type of error.
        
        Returns:
            List of debug suggestions.
        """
        suggestions: List[DebugSuggestion] = []
        
        # Parse location
        if ":" in error_location:
            file_path, line_str = error_location.rsplit(":", 1)
            try:
                line_number = int(line_str)
            except ValueError:
                return suggestions
        else:
            return suggestions
        
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return suggestions
        
        # Read file and analyze
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            
            if 0 <= line_number - 1 < len(lines):
                error_line = lines[line_number - 1]
                
                # Extract variables from error line
                variables = self._extract_variables(error_line)
                
                # Suggest logging
                for var in variables[:3]:  # Limit to 3 variables
                    log_code = f"print(f'DEBUG: {var} = {{{var}}}')"
                    suggestions.append(DebugSuggestion(
                        location=DebugPoint(
                            file_path=file_path,
                            line_number=line_number - 1,  # Before error line
                            variable_names=[var],
                        ),
                        suggestion_type="log",
                        code=log_code,
                        explanation=f"Add logging to inspect '{var}' value",
                    ))
                
                # Suggest breakpoint
                suggestions.append(DebugSuggestion(
                    location=DebugPoint(
                        file_path=file_path,
                        line_number=line_number,
                        variable_names=variables,
                    ),
                    suggestion_type="breakpoint",
                    code=f"import pdb; pdb.set_trace()  # Debug at line {line_number}",
                    explanation="Add breakpoint to inspect state",
                ))
        
        except Exception:
            pass
        
        return suggestions
    
    def _extract_variables(self, code_line: str) -> List[str]:
        """Extract variable names from code line."""
        variables: List[str] = []
        
        # Simple pattern matching for variable names
        # Match word characters that look like variables
        var_pattern = r'\b([a-z_][a-z0-9_]*)\b'
        matches = re.findall(var_pattern, code_line, re.IGNORECASE)
        
        # Filter out keywords
        keywords = {
            "if", "else", "for", "while", "def", "class", "import", "from",
            "return", "print", "pass", "break", "continue", "try", "except",
            "finally", "with", "as", "in", "is", "not", "and", "or", "True",
            "False", "None", "self", "NoneType",
        }
        
        variables = [m for m in matches if m.lower() not in keywords]
        
        return list(set(variables))  # Remove duplicates
    
    def add_debug_logging(
        self,
        file_path: str,
        line_number: int,
        variables: List[str],
    ) -> str:
        """
        Generate debug logging code.
        
        Args:
            file_path: File path.
            line_number: Line number to add logging before.
            variables: Variables to log.
        
        Returns:
            Debug logging code.
        """
        if not variables:
            return ""
        
        log_lines = []
        for var in variables:
            log_lines.append(f"    print(f'DEBUG: {var} = {{{var}}}')")
        
        return "\n".join(log_lines)
    
    def generate_debug_code(
        self,
        error_analysis,
    ) -> str:
        """
        Generate debug code based on error analysis.
        
        Args:
            error_analysis: ErrorAnalysis object.
        
        Returns:
            Debug code to add.
        """
        debug_code = []
        
        if error_analysis.error_context.root_cause:
            loc = error_analysis.error_context.root_cause
            
            debug_code.append(f"# Debug at {loc.file_path}:{loc.line_number}")
            debug_code.append(f"# Error: {error_analysis.error_context.error_type}")
            
            if loc.code_snippet:
                debug_code.append(f"# Code: {loc.code_snippet.strip()}")
            
            # Add logging suggestions
            if loc.code_snippet:
                variables = self._extract_variables(loc.code_snippet)
                for var in variables[:3]:
                    debug_code.append(f"print(f'DEBUG: {var} = {{{var}}}')")
        
        return "\n".join(debug_code)


def create_debugger(repo_root: Path) -> Debugger:
    """
    Create debugger instance.
    
    Args:
        repo_root: Repository root directory.
    
    Returns:
        Debugger instance.
    """
    return Debugger(repo_root)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    debugger = create_debugger(repo_root)
    
    suggestions = debugger.suggest_debug_points("test.py:10", "NameError")
    for suggestion in suggestions:
        print(f"{suggestion.suggestion_type}: {suggestion.code}")
