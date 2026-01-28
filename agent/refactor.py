"""Multi-file refactoring tools - Claude Code level."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import re

try:
    from memory.knowledge_graph import build_codebase_graph, CodebaseGraph
    from memory.call_graph import build_call_graph, CallGraphBuilder
    from memory.symbols import extract_symbols, find_all_usages
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from memory.knowledge_graph import build_codebase_graph, CodebaseGraph
        from memory.call_graph import build_call_graph, CallGraphBuilder
        from memory.symbols import extract_symbols, find_all_usages
    except ImportError:
        CodebaseGraph = None
        CallGraphBuilder = None
        build_codebase_graph = None
        build_call_graph = None
        find_all_usages = None


@dataclass
class RefactoringPlan:
    """Plan for a refactoring operation."""
    
    operation: str  # 'rename', 'extract', 'inline', 'move'
    target_symbol: str
    new_name: Optional[str] = None
    affected_files: List[str] = field(default_factory=list)
    changes: List[Dict[str, str]] = field(default_factory=list)  # List of {file, old_code, new_code}
    risks: List[str] = field(default_factory=list)


class RefactoringTool:
    """Safe multi-file refactoring."""
    
    def __init__(self, repo_root: Path):
        """
        Initialize refactoring tool.
        
        Args:
            repo_root: Repository root directory.
        """
        self.repo_root = Path(repo_root).resolve()
        self.knowledge_graph: Optional[CodebaseGraph] = None
        self.call_graph: Optional[CallGraphBuilder] = None
        self._initialize_graphs()
    
    def _initialize_graphs(self) -> None:
        """Initialize knowledge and call graphs."""
        if build_codebase_graph:
            try:
                self.knowledge_graph = build_codebase_graph(self.repo_root)
                if build_call_graph and self.knowledge_graph:
                    self.call_graph = build_call_graph(self.repo_root, self.knowledge_graph)
            except Exception:
                pass
    
    def rename_symbol(
        self,
        old_name: str,
        new_name: str,
        symbol_type: str = "function",  # 'function', 'class', 'variable'
        file_path: Optional[str] = None,
    ) -> RefactoringPlan:
        """
        Plan a symbol rename across all files.
        
        Args:
            old_name: Old symbol name.
            new_name: New symbol name.
            symbol_type: Type of symbol ('function', 'class', 'variable').
            file_path: Optional specific file to rename in.
        
        Returns:
            RefactoringPlan with all changes needed.
        """
        plan = RefactoringPlan(
            operation="rename",
            target_symbol=old_name,
            new_name=new_name,
        )
        
        # Find all usages
        if self.knowledge_graph:
            usages = self.knowledge_graph.find_all_usages(old_name)
            affected_files = list(set(loc.file_path for loc in usages))
        elif find_all_usages:
            affected_files = find_all_usages(old_name, self.repo_root)
        else:
            # Fallback: search files manually
            affected_files = self._find_files_with_symbol(old_name)
        
        plan.affected_files = affected_files
        
        # Generate changes for each file
        for file_path in affected_files:
            change = self._generate_rename_change(file_path, old_name, new_name, symbol_type)
            if change:
                plan.changes.append(change)
        
        # Assess risks
        plan.risks = self._assess_rename_risks(old_name, new_name, affected_files)
        
        return plan
    
    def _find_files_with_symbol(self, symbol_name: str) -> List[str]:
        """Find files containing a symbol (fallback method)."""
        from memory.index import scan_repo
        
        files: List[str] = []
        repo_files = scan_repo(str(self.repo_root))
        
        for file_path in repo_files:
            if not file_path.endswith(".py"):
                continue
            
            full_path = self.repo_root / file_path
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                if symbol_name in content:
                    files.append(file_path)
            except Exception:
                continue
        
        return files
    
    def _generate_rename_change(
        self,
        file_path: str,
        old_name: str,
        new_name: str,
        symbol_type: str,
    ) -> Optional[Dict[str, str]]:
        """Generate rename change for a file."""
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return None
        
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        
        # Simple rename (can be enhanced with AST)
        # Replace whole-word matches
        pattern = r'\b' + re.escape(old_name) + r'\b'
        new_content = re.sub(pattern, new_name, content)
        
        if new_content != content:
            return {
                "file": file_path,
                "old_code": content,
                "new_code": new_content,
                "change_type": "rename",
            }
        
        return None
    
    def _assess_rename_risks(
        self,
        old_name: str,
        new_name: str,
        affected_files: List[str],
    ) -> List[str]:
        """Assess risks of renaming."""
        risks = []
        
        if len(affected_files) > 10:
            risks.append(f"Many files affected ({len(affected_files)}) - review carefully")
        
        # Check if new name conflicts with existing symbols
        if self.knowledge_graph:
            existing = self.knowledge_graph.find_all_usages(new_name)
            if existing:
                risks.append(f"New name '{new_name}' already exists in codebase")
        
        # Check if symbol is part of public API
        if any("__init__.py" in f for f in affected_files):
            risks.append("Symbol may be part of public API - check imports")
        
        return risks
    
    def extract_function(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        new_function_name: str,
    ) -> RefactoringPlan:
        """
        Plan extraction of code into a new function.
        
        Args:
            file_path: File containing code to extract.
            start_line: Start line of code to extract.
            end_line: End line of code to extract.
            new_function_name: Name for new function.
        
        Returns:
            RefactoringPlan with extraction changes.
        """
        plan = RefactoringPlan(
            operation="extract",
            target_symbol=new_function_name,
        )
        
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return plan
        
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            
            # Extract code block
            extracted_code = "\n".join(lines[start_line - 1:end_line])
            
            # Generate function signature (simplified)
            # In production, would analyze AST to determine parameters
            function_code = f"def {new_function_name}():\n"
            function_code += "    " + "\n    ".join(extracted_code.splitlines())
            
            # Create new content with function extracted
            before = "\n".join(lines[:start_line - 1])
            after = "\n".join(lines[end_line:])
            call_site = f"{new_function_name}()"
            
            new_content = "\n".join([
                before,
                function_code,
                "",
                call_site,
                after,
            ])
            
            plan.changes.append({
                "file": file_path,
                "old_code": content,
                "new_code": new_content,
                "change_type": "extract",
            })
            plan.affected_files = [file_path]
        
        except Exception as e:
            plan.risks.append(f"Extraction failed: {e}")
        
        return plan
    
    def inline_function(
        self,
        function_name: str,
        file_path: str,
        call_site_line: int,
    ) -> RefactoringPlan:
        """
        Plan inlining a function call.
        
        Args:
            function_name: Name of function to inline.
            file_path: File containing the call site.
            call_site_line: Line number of function call.
        
        Returns:
            RefactoringPlan with inlining changes.
        """
        plan = RefactoringPlan(
            operation="inline",
            target_symbol=function_name,
        )
        
        # Find function definition
        if self.knowledge_graph:
            func_info = self.knowledge_graph.get_function_info(function_name, file_path)
            if func_info:
                # Get function body
                full_path = self.repo_root / file_path
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()
                    
                    # Extract function body (simplified)
                    func_start = func_info.line - 1
                    func_end = func_start + 10  # Simplified - would parse AST properly
                    func_body = "\n".join(lines[func_start:func_end])
                    
                    # Replace call with function body
                    call_line = lines[call_site_line - 1]
                    new_content = content.replace(call_line, func_body)
                    
                    plan.changes.append({
                        "file": file_path,
                        "old_code": content,
                        "new_code": new_content,
                        "change_type": "inline",
                    })
                    plan.affected_files = [file_path]
                except Exception:
                    plan.risks.append("Failed to inline function")
        
        return plan
    
    def validate_refactoring(self, plan: RefactoringPlan) -> Tuple[bool, List[str]]:
        """
        Validate a refactoring plan.
        
        Args:
            plan: Refactoring plan to validate.
        
        Returns:
            Tuple of (is_valid, warnings).
        """
        warnings: List[str] = []
        
        # Check if all affected files exist
        for file_path in plan.affected_files:
            full_path = self.repo_root / file_path
            if not full_path.exists():
                warnings.append(f"File not found: {file_path}")
        
        # Check for conflicts
        if plan.operation == "rename" and plan.new_name:
            if self.knowledge_graph:
                existing = self.knowledge_graph.find_all_usages(plan.new_name)
                if existing:
                    warnings.append(f"New name '{plan.new_name}' conflicts with existing symbols")
        
        # Check risks
        warnings.extend(plan.risks)
        
        is_valid = len([w for w in warnings if "conflict" in w.lower()]) == 0
        
        return is_valid, warnings


def create_refactoring_tool(repo_root: Path) -> RefactoringTool:
    """
    Create a refactoring tool instance.
    
    Args:
        repo_root: Repository root directory.
    
    Returns:
        RefactoringTool instance.
    """
    return RefactoringTool(repo_root)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    tool = create_refactoring_tool(repo_root)
    
    # Plan rename
    plan = tool.rename_symbol("old_function", "new_function", "function")
    print(f"Rename plan: {len(plan.affected_files)} files affected")
    print(f"Risks: {plan.risks}")
