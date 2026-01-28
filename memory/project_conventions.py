"""Project conventions learning and enforcement - Claude Code level."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
import ast
import re

try:
    from .symbols import extract_symbols, FileSymbols
    from .index import scan_repo
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from memory.symbols import extract_symbols, FileSymbols
    from memory.index import scan_repo


@dataclass
class CodeStyle:
    """Detected code style conventions."""
    
    quote_style: str = "double"  # 'single' or 'double'
    max_line_length: int = 88  # Common default
    indent_style: str = "spaces"  # 'spaces' or 'tabs'
    indent_size: int = 4
    import_order: List[str] = field(default_factory=lambda: ["stdlib", "third_party", "local"])
    trailing_commas: bool = True
    blank_lines_before_class: int = 2
    blank_lines_before_function: int = 2


@dataclass
class NamingConventions:
    """Detected naming conventions."""
    
    function_naming: str = "snake_case"  # 'snake_case', 'camelCase', 'PascalCase'
    class_naming: str = "PascalCase"
    constant_naming: str = "UPPER_SNAKE_CASE"
    variable_naming: str = "snake_case"
    private_prefix: str = "_"  # '_' for private


@dataclass
class ProjectConventions:
    """Complete project conventions."""
    
    style: CodeStyle = field(default_factory=CodeStyle)
    naming: NamingConventions = field(default_factory=NamingConventions)
    patterns: Dict[str, List[str]] = field(default_factory=dict)  # Common patterns
    file_structure: Dict[str, List[str]] = field(default_factory=dict)  # Common file structures


class ProjectConventionLearner:
    """Learns coding conventions from existing codebase."""
    
    def __init__(self, repo_root: Path):
        """
        Initialize convention learner.
        
        Args:
            repo_root: Repository root directory.
        """
        self.repo_root = Path(repo_root).resolve()
        self.conventions = ProjectConventions()
        self._analyzed_files: Set[str] = set()
    
    def learn_from_codebase(self, sample_size: int = 50) -> ProjectConventions:
        """
        Learn conventions from codebase.
        
        Args:
            sample_size: Number of files to analyze.
        
        Returns:
            Learned conventions.
        """
        repo_files = scan_repo(str(self.repo_root))
        python_files = [f for f in repo_files if f.endswith(".py")][:sample_size]
        
        quote_styles: List[str] = []
        line_lengths: List[int] = []
        indent_styles: List[str] = []
        indent_sizes: List[int] = []
        function_names: List[str] = []
        class_names: List[str] = []
        constant_names: List[str] = []
        
        for file_path in python_files:
            try:
                full_path = self.repo_root / file_path
                content = full_path.read_text(encoding="utf-8", errors="replace")
                
                # Analyze style
                style_info = self._analyze_style(content)
                if style_info["quote_style"]:
                    quote_styles.append(style_info["quote_style"])
                if style_info["indent_style"]:
                    indent_styles.append(style_info["indent_style"])
                if style_info["indent_size"]:
                    indent_sizes.append(style_info["indent_size"])
                
                # Analyze line lengths
                for line in content.splitlines():
                    if line.strip() and not line.strip().startswith("#"):
                        line_lengths.append(len(line))
                
                # Analyze naming
                file_symbols = extract_symbols(full_path, use_cache=True)
                for symbol in file_symbols.symbols:
                    if symbol.kind == "function":
                        function_names.append(symbol.name)
                    elif symbol.kind == "class":
                        class_names.append(symbol.name)
                    elif symbol.name.isupper() and "_" in symbol.name:
                        constant_names.append(symbol.name)
                
            except Exception:
                continue
        
        # Determine conventions
        self.conventions.style.quote_style = Counter(quote_styles).most_common(1)[0][0] if quote_styles else "double"
        self.conventions.style.indent_style = Counter(indent_styles).most_common(1)[0][0] if indent_styles else "spaces"
        self.conventions.style.indent_size = int(sum(indent_sizes) / len(indent_sizes)) if indent_sizes else 4
        self.conventions.style.max_line_length = int(sum(line_lengths) / len(line_lengths)) if line_lengths else 88
        
        # Determine naming conventions
        self.conventions.naming.function_naming = self._detect_naming_style(function_names)
        self.conventions.naming.class_naming = self._detect_naming_style(class_names)
        self.conventions.naming.constant_naming = self._detect_naming_style(constant_names)
        
        # Learn common patterns
        self._learn_patterns(python_files[:20])  # Analyze first 20 files for patterns
        
        return self.conventions
    
    def _analyze_style(self, content: str) -> Dict[str, Optional[str | int]]:
        """Analyze code style from content."""
        style_info: Dict[str, Optional[str | int]] = {
            "quote_style": None,
            "indent_style": None,
            "indent_size": None,
        }
        
        # Detect quote style
        single_quotes = content.count("'")
        double_quotes = content.count('"')
        if single_quotes > double_quotes * 1.5:
            style_info["quote_style"] = "single"
        elif double_quotes > single_quotes * 1.5:
            style_info["quote_style"] = "double"
        
        # Detect indent style
        lines = content.splitlines()
        for line in lines[:50]:  # Check first 50 lines
            if line.startswith(" "):
                style_info["indent_style"] = "spaces"
                # Count spaces at start
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    style_info["indent_size"] = indent
                break
            elif line.startswith("\t"):
                style_info["indent_style"] = "tabs"
                break
        
        return style_info
    
    def _detect_naming_style(self, names: List[str]) -> str:
        """Detect naming style from list of names."""
        if not names:
            return "snake_case"
        
        snake_case_count = sum(1 for n in names if "_" in n and n.islower())
        camel_case_count = sum(1 for n in names if n[0].islower() and not "_" in n and any(c.isupper() for c in n))
        pascal_case_count = sum(1 for n in names if n[0].isupper() and not "_" in n)
        
        total = len(names)
        if snake_case_count / total > 0.7:
            return "snake_case"
        elif camel_case_count / total > 0.7:
            return "camelCase"
        elif pascal_case_count / total > 0.7:
            return "PascalCase"
        else:
            return "snake_case"  # Default
    
    def _learn_patterns(self, files: List[str]) -> None:
        """Learn common code patterns."""
        patterns: Dict[str, List[str]] = {
            "imports": [],
            "decorators": [],
            "error_handling": [],
        }
        
        for file_path in files:
            try:
                full_path = self.repo_root / file_path
                content = full_path.read_text(encoding="utf-8", errors="replace")
                
                # Extract import patterns
                import_lines = [line for line in content.splitlines() if line.strip().startswith("import") or line.strip().startswith("from")]
                patterns["imports"].extend(import_lines[:10])  # Limit per file
                
                # Extract decorator patterns
                decorator_lines = [line for line in content.splitlines() if line.strip().startswith("@")]
                patterns["decorators"].extend(decorator_lines[:5])
                
                # Extract error handling patterns
                if "try:" in content:
                    patterns["error_handling"].append("try-except")
                if "raise" in content:
                    patterns["error_handling"].append("raise-exception")
                
            except Exception:
                continue
        
        self.conventions.patterns = patterns
    
    def enforce_conventions(self, code: str) -> Tuple[str, List[str]]:
        """
        Enforce conventions on code.
        
        Args:
            code: Code to enforce conventions on.
        
        Returns:
            Tuple of (enforced_code, violations_found).
        """
        violations: List[str] = []
        enforced = code
        
        # Check quote style
        if self.conventions.style.quote_style == "double":
            # Convert single quotes to double (simple heuristic)
            single_quote_count = code.count("'")
            double_quote_count = code.count('"')
            if single_quote_count > double_quote_count:
                violations.append("Using single quotes instead of double quotes")
        elif self.conventions.style.quote_style == "single":
            double_quote_count = code.count('"')
            single_quote_count = code.count("'")
            if double_quote_count > single_quote_count:
                violations.append("Using double quotes instead of single quotes")
        
        # Check line length
        lines = code.splitlines()
        long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > self.conventions.style.max_line_length]
        if long_lines:
            violations.append(f"Lines {long_lines[:5]} exceed max length ({self.conventions.style.max_line_length})")
        
        # Check naming conventions
        violations.extend(self._check_naming(code))
        
        return enforced, violations
    
    def _check_naming(self, code: str) -> List[str]:
        """Check naming conventions."""
        violations: List[str] = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    name = node.name
                    expected_style = self.conventions.naming.function_naming
                    if not self._matches_naming_style(name, expected_style):
                        violations.append(f"Function '{name}' doesn't match {expected_style} convention")
                
                elif isinstance(node, ast.ClassDef):
                    name = node.name
                    expected_style = self.conventions.naming.class_naming
                    if not self._matches_naming_style(name, expected_style):
                        violations.append(f"Class '{name}' doesn't match {expected_style} convention")
        
        except SyntaxError:
            pass  # Can't check naming if code is invalid
        
        return violations
    
    def _matches_naming_style(self, name: str, style: str) -> bool:
        """Check if name matches naming style."""
        if style == "snake_case":
            return "_" in name or name.islower()
        elif style == "camelCase":
            return name[0].islower() and not "_" in name
        elif style == "PascalCase":
            return name[0].isupper() and not "_" in name
        elif style == "UPPER_SNAKE_CASE":
            return name.isupper() and "_" in name
        return True  # Default: accept any
    
    def load_rules(self, rules_file: Path) -> bool:
        """
        Load conventions from .lca-rules.yaml file.
        
        Args:
            rules_file: Path to rules file.
        
        Returns:
            True if loaded successfully.
        """
        if not rules_file.exists():
            return False
        
        try:
            import yaml
            with rules_file.open("r") as f:
                rules = yaml.safe_load(f)
            
            conventions = rules.get("conventions", {})
            
            # Apply rules
            if "use_double_quotes" in conventions:
                self.conventions.style.quote_style = "double" if conventions["use_double_quotes"] else "single"
            if "max_line_length" in conventions:
                self.conventions.style.max_line_length = conventions["max_line_length"]
            if "import_order" in conventions:
                self.conventions.style.import_order = conventions["import_order"]
            
            return True
        except Exception:
            return False
    
    def save_rules(self, rules_file: Path) -> bool:
        """
        Save conventions to .lca-rules.yaml file.
        
        Args:
            rules_file: Path to save rules file.
        
        Returns:
            True if saved successfully.
        """
        try:
            import yaml
            
            rules = {
                "conventions": {
                    "use_double_quotes": self.conventions.style.quote_style == "double",
                    "max_line_length": self.conventions.style.max_line_length,
                    "indent_size": self.conventions.style.indent_size,
                    "indent_style": self.conventions.style.indent_style,
                    "import_order": self.conventions.style.import_order,
                    "function_naming": self.conventions.naming.function_naming,
                    "class_naming": self.conventions.naming.class_naming,
                }
            }
            
            rules_file.parent.mkdir(parents=True, exist_ok=True)
            with rules_file.open("w") as f:
                yaml.dump(rules, f, default_flow_style=False)
            
            return True
        except Exception:
            return False


def learn_project_conventions(repo_root: Path, sample_size: int = 50) -> ProjectConventions:
    """
    Learn conventions from a codebase.
    
    Args:
        repo_root: Repository root directory.
        sample_size: Number of files to analyze.
    
    Returns:
        Learned conventions.
    """
    learner = ProjectConventionLearner(repo_root)
    return learner.learn_from_codebase(sample_size)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    conventions = learn_project_conventions(repo_root)
    
    print("Learned Conventions:")
    print(f"  Quote style: {conventions.style.quote_style}")
    print(f"  Max line length: {conventions.style.max_line_length}")
    print(f"  Indent: {conventions.style.indent_size} {conventions.style.indent_style}")
    print(f"  Function naming: {conventions.naming.function_naming}")
    print(f"  Class naming: {conventions.naming.class_naming}")
