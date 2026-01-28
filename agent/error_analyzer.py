"""Error analysis - Claude Code level error understanding and debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import traceback


@dataclass
class ErrorLocation:
    """Location of an error."""
    
    file_path: str
    line_number: int
    column_number: Optional[int] = None
    function_name: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class ErrorContext:
    """Context around an error."""
    
    error_type: str
    error_message: str
    stack_trace: List[ErrorLocation] = field(default_factory=list)
    root_cause: Optional[ErrorLocation] = None
    related_files: List[str] = field(default_factory=list)


@dataclass
class FixSuggestion:
    """Suggestion for fixing an error."""
    
    description: str
    confidence: float  # 0.0 to 1.0
    code_change: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class ErrorAnalysis:
    """Complete error analysis."""
    
    error_context: ErrorContext
    root_cause: Optional[str] = None
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)
    similar_errors: List[str] = field(default_factory=list)
    debugging_steps: List[str] = field(default_factory=list)


class ErrorAnalyzer:
    """Deep error analysis and debugging."""
    
    def __init__(self, repo_root: Path):
        """
        Initialize error analyzer.
        
        Args:
            repo_root: Repository root directory.
        """
        self.repo_root = Path(repo_root).resolve()
        self.error_patterns: Dict[str, List[str]] = {}
        self._load_error_patterns()
    
    def _load_error_patterns(self) -> None:
        """Load common error patterns."""
        self.error_patterns = {
            "ImportError": [
                "Module not found",
                "No module named",
                "cannot import name",
            ],
            "NameError": [
                "name is not defined",
                "undefined name",
            ],
            "TypeError": [
                "unsupported operand",
                "takes X positional arguments",
                "object is not callable",
            ],
            "AttributeError": [
                "object has no attribute",
                "'NoneType' object has no attribute",
            ],
            "KeyError": [
                "key not found",
            ],
            "IndexError": [
                "list index out of range",
            ],
            "ValueError": [
                "invalid literal",
                "unexpected value",
            ],
            "SyntaxError": [
                "invalid syntax",
                "unexpected EOF",
            ],
        }
    
    def analyze_error(
        self,
        error_text: str,
        stack_trace: Optional[str] = None,
    ) -> ErrorAnalysis:
        """
        Analyze an error and provide insights.
        
        Args:
            error_text: Error message or full traceback.
            stack_trace: Optional separate stack trace.
        
        Returns:
            ErrorAnalysis with insights and suggestions.
        """
        # Parse error
        error_context = self._parse_error(error_text, stack_trace)
        
        # Analyze root cause
        root_cause = self._identify_root_cause(error_context)
        
        # Generate fix suggestions
        fix_suggestions = self._generate_fix_suggestions(error_context)
        
        # Find similar errors
        similar_errors = self._find_similar_errors(error_context)
        
        # Generate debugging steps
        debugging_steps = self._generate_debugging_steps(error_context)
        
        return ErrorAnalysis(
            error_context=error_context,
            root_cause=root_cause,
            fix_suggestions=fix_suggestions,
            similar_errors=similar_errors,
            debugging_steps=debugging_steps,
        )
    
    def _parse_error(
        self,
        error_text: str,
        stack_trace: Optional[str],
    ) -> ErrorContext:
        """Parse error text into ErrorContext."""
        # Extract error type and message
        error_type = "UnknownError"
        error_message = error_text
        
        # Try to extract from traceback format
        traceback_pattern = r"(\w+Error|Exception):\s*(.+)"
        match = re.search(traceback_pattern, error_text)
        if match:
            error_type = match.group(1)
            error_message = match.group(2)
        
        # Parse stack trace
        locations = self._parse_stack_trace(error_text + (stack_trace or ""))
        
        # Find root cause location (usually the last one in stack)
        root_cause = locations[-1] if locations else None
        
        # Find related files
        related_files = list(set(loc.file_path for loc in locations))
        
        return ErrorContext(
            error_type=error_type,
            error_message=error_message,
            stack_trace=locations,
            root_cause=root_cause,
            related_files=related_files,
        )
    
    def _parse_stack_trace(self, traceback_text: str) -> List[ErrorLocation]:
        """Parse stack trace into ErrorLocation objects."""
        locations: List[ErrorLocation] = []
        
        # Pattern: File "path/to/file.py", line X, in function_name
        pattern = r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+(\w+))?'
        matches = re.findall(pattern, traceback_text)
        
        for file_path, line_str, function_name in matches:
            try:
                line_number = int(line_str)
                # Make path relative to repo root
                full_path = Path(file_path)
                try:
                    rel_path = str(full_path.relative_to(self.repo_root))
                except ValueError:
                    rel_path = file_path
                
                # Try to read code snippet
                code_snippet = None
                if full_path.exists():
                    try:
                        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if 0 <= line_number - 1 < len(lines):
                            code_snippet = lines[line_number - 1]
                    except Exception:
                        pass
                
                locations.append(ErrorLocation(
                    file_path=rel_path,
                    line_number=line_number,
                    function_name=function_name or None,
                    code_snippet=code_snippet,
                ))
            except ValueError:
                continue
        
        return locations
    
    def _identify_root_cause(self, context: ErrorContext) -> Optional[str]:
        """Identify root cause of error."""
        error_type = context.error_type
        error_message = context.error_message.lower()
        
        # Common root causes
        if error_type == "ImportError":
            if "no module named" in error_message:
                module_match = re.search(r"no module named ['\"](\w+)['\"]", error_message)
                if module_match:
                    return f"Missing module: {module_match.group(1)}. Install it or check import path."
            return "Import error: Check module name and installation."
        
        elif error_type == "NameError":
            name_match = re.search(r"name ['\"](\w+)['\"] is not defined", error_message)
            if name_match:
                return f"Undefined variable/function: {name_match.group(1)}. Check spelling or define it."
            return "Name error: Variable or function not defined."
        
        elif error_type == "TypeError":
            if "takes" in error_message and "arguments" in error_message:
                return "Type error: Function called with wrong number of arguments."
            if "not callable" in error_message:
                return "Type error: Trying to call something that's not a function."
            return "Type error: Wrong type used in operation."
        
        elif error_type == "AttributeError":
            attr_match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_message)
            if attr_match:
                obj_type = attr_match.group(1)
                attr_name = attr_match.group(2)
                return f"Attribute error: {obj_type} doesn't have attribute '{attr_name}'. Check object type."
            return "Attribute error: Object doesn't have the requested attribute."
        
        elif error_type == "KeyError":
            key_match = re.search(r"['\"](\w+)['\"]", error_message)
            if key_match:
                return f"Key error: Key '{key_match.group(1)}' not found in dictionary."
            return "Key error: Dictionary key not found."
        
        elif error_type == "IndexError":
            return "Index error: List/array index out of range. Check list length before accessing."
        
        elif error_type == "ValueError":
            return "Value error: Invalid value passed to function. Check input format."
        
        elif error_type == "SyntaxError":
            return "Syntax error: Invalid Python syntax. Check brackets, quotes, and indentation."
        
        return None
    
    def _generate_fix_suggestions(self, context: ErrorContext) -> List[FixSuggestion]:
        """Generate fix suggestions."""
        suggestions: List[FixSuggestion] = []
        error_type = context.error_type
        error_message = context.error_message.lower()
        
        if error_type == "ImportError":
            module_match = re.search(r"no module named ['\"](\w+)['\"]", error_message)
            if module_match:
                module_name = module_match.group(1)
                suggestions.append(FixSuggestion(
                    description=f"Install missing module: pip install {module_name}",
                    confidence=0.9,
                    explanation=f"The module '{module_name}' is not installed.",
                ))
            suggestions.append(FixSuggestion(
                description="Check import path and module name spelling",
                confidence=0.7,
                explanation="Verify the import statement matches the actual module name.",
            ))
        
        elif error_type == "NameError":
            name_match = re.search(r"name ['\"](\w+)['\"] is not defined", error_message)
            if name_match:
                var_name = name_match.group(1)
                suggestions.append(FixSuggestion(
                    description=f"Define variable '{var_name}' before use",
                    confidence=0.9,
                    explanation=f"The variable '{var_name}' is used before being defined.",
                ))
            suggestions.append(FixSuggestion(
                description="Check variable spelling and scope",
                confidence=0.7,
                explanation="Verify variable name is spelled correctly and is in scope.",
            ))
        
        elif error_type == "TypeError":
            if "takes" in error_message and "arguments" in error_message:
                suggestions.append(FixSuggestion(
                    description="Check function signature and argument count",
                    confidence=0.8,
                    explanation="Function called with wrong number of arguments.",
                ))
            if "not callable" in error_message:
                suggestions.append(FixSuggestion(
                    description="Verify object is a function before calling",
                    confidence=0.8,
                    explanation="Trying to call something that's not callable.",
                ))
        
        elif error_type == "AttributeError":
            attr_match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_message)
            if attr_match:
                obj_type = attr_match.group(1)
                attr_name = attr_match.group(2)
                suggestions.append(FixSuggestion(
                    description=f"Check if {obj_type} has method '{attr_name}' or use correct method name",
                    confidence=0.8,
                    explanation=f"{obj_type} objects don't have '{attr_name}' attribute.",
                ))
        
        elif error_type == "KeyError":
            key_match = re.search(r"['\"](\w+)['\"]", error_message)
            if key_match:
                key_name = key_match.group(1)
                suggestions.append(FixSuggestion(
                    description=f"Check if key '{key_name}' exists before accessing: use .get() or 'in' check",
                    confidence=0.9,
                    code_change=f"# Instead of: dict['{key_name}']\n# Use: dict.get('{key_name}') or '{key_name}' in dict",
                    explanation=f"Key '{key_name}' doesn't exist in dictionary.",
                ))
        
        elif error_type == "IndexError":
            suggestions.append(FixSuggestion(
                description="Check list length before accessing: use len() check or try/except",
                confidence=0.9,
                code_change="# Before accessing: if len(list) > index:\n#     value = list[index]",
                explanation="List index out of range.",
            ))
        
        elif error_type == "SyntaxError":
            suggestions.append(FixSuggestion(
                description="Check brackets, parentheses, quotes, and indentation",
                confidence=0.8,
                explanation="Invalid Python syntax detected.",
            ))
        
        # Add general debugging suggestion
        if context.root_cause:
            suggestions.append(FixSuggestion(
                description=f"Add debug logging at line {context.root_cause.line_number} in {context.root_cause.file_path}",
                confidence=0.6,
                explanation="Add print/logging statements to inspect values at error location.",
            ))
        
        return suggestions
    
    def _find_similar_errors(self, context: ErrorContext) -> List[str]:
        """Find similar errors in codebase."""
        # This would search for similar error patterns in code
        # For now, return empty list
        return []
    
    def _generate_debugging_steps(self, context: ErrorContext) -> List[str]:
        """Generate debugging steps."""
        steps: List[str] = []
        
        if context.root_cause:
            steps.append(f"1. Check line {context.root_cause.line_number} in {context.root_cause.file_path}")
            if context.root_cause.code_snippet:
                steps.append(f"   Code: {context.root_cause.code_snippet.strip()}")
        
        steps.append(f"2. Error type: {context.error_type}")
        steps.append(f"3. Error message: {context.error_message}")
        
        if context.stack_trace:
            steps.append(f"4. Stack trace has {len(context.stack_trace)} frames")
            if len(context.stack_trace) > 1:
                steps.append(f"5. Check calling function: {context.stack_trace[-2].function_name}")
        
        steps.append("6. Add debug logging to inspect variable values")
        steps.append("7. Check related files: " + ", ".join(context.related_files[:3]))
        
        return steps
    
    def explain_error(self, error_text: str) -> str:
        """
        Provide human-readable error explanation.
        
        Args:
            error_text: Error text.
        
        Returns:
            Human-readable explanation.
        """
        analysis = self.analyze_error(error_text)
        
        explanation = f"**Error Type**: {analysis.error_context.error_type}\n\n"
        explanation += f"**Message**: {analysis.error_context.error_message}\n\n"
        
        if analysis.root_cause:
            explanation += f"**Root Cause**: {analysis.root_cause}\n\n"
        
        if analysis.error_context.root_cause:
            loc = analysis.error_context.root_cause
            explanation += f"**Location**: {loc.file_path}:{loc.line_number}\n"
            if loc.function_name:
                explanation += f"**Function**: {loc.function_name}\n"
            explanation += "\n"
        
        if analysis.fix_suggestions:
            explanation += "**Fix Suggestions**:\n"
            for i, suggestion in enumerate(analysis.fix_suggestions[:3], 1):
                explanation += f"{i}. {suggestion.description}\n"
                if suggestion.explanation:
                    explanation += f"   {suggestion.explanation}\n"
            explanation += "\n"
        
        if analysis.debugging_steps:
            explanation += "**Debugging Steps**:\n"
            for step in analysis.debugging_steps[:5]:
                explanation += f"{step}\n"
        
        return explanation


def create_error_analyzer(repo_root: Path) -> ErrorAnalyzer:
    """
    Create error analyzer instance.
    
    Args:
        repo_root: Repository root directory.
    
    Returns:
        ErrorAnalyzer instance.
    """
    return ErrorAnalyzer(repo_root)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    analyzer = create_error_analyzer(repo_root)
    
    error_text = sys.argv[2] if len(sys.argv) > 2 else "NameError: name 'x' is not defined"
    explanation = analyzer.explain_error(error_text)
    print(explanation)
