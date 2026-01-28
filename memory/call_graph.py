"""Call graph analysis - who calls what."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

try:
    import ast
    from .symbols import extract_symbols, Symbol
    from .knowledge_graph import CodebaseGraph, CodeLocation
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    import ast
    from memory.symbols import extract_symbols, Symbol
    from memory.knowledge_graph import CodebaseGraph, CodeLocation


@dataclass
class CallSite:
    """A call site (where a function is called)."""
    
    caller_file: str
    caller_line: int
    caller_function: Optional[str] = None
    callee_name: str = ""
    callee_file: Optional[str] = None


@dataclass
class CallGraph:
    """Call graph for the codebase."""
    
    calls: Dict[str, List[CallSite]] = field(default_factory=dict)  # callee -> call sites
    callers: Dict[str, List[str]] = field(default_factory=dict)  # callee -> caller functions
    callees: Dict[str, List[str]] = field(default_factory=dict)  # caller -> callee functions


class CallGraphBuilder:
    """Builds call graph by analyzing AST."""
    
    def __init__(self, repo_root: Path, codebase_graph: Optional[CodebaseGraph] = None):
        """
        Initialize call graph builder.
        
        Args:
            repo_root: Repository root directory.
            codebase_graph: Optional codebase graph for symbol resolution.
        """
        self.repo_root = Path(repo_root).resolve()
        self.codebase_graph = codebase_graph
        self.call_graph = CallGraph()
        self._build_call_graph()
    
    def _build_call_graph(self) -> None:
        """Build call graph by analyzing all Python files."""
        from memory.index import scan_repo
        
        repo_files = scan_repo(str(self.repo_root))
        python_files = [f for f in repo_files if f.endswith(".py")]
        
        for file_path in python_files:
            self._analyze_file(file_path)
    
    def _analyze_file(self, file_path: str) -> None:
        """Analyze a file for function calls."""
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return
        
        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=file_path)
        except Exception:
            return
        
        # Extract symbols from file
        file_symbols = extract_symbols(full_path, use_cache=True)
        
        # Build function name to line mapping
        function_lines: Dict[int, str] = {}
        for symbol in file_symbols.symbols:
            if symbol.kind in ("function", "method"):
                function_lines[symbol.line] = symbol.name
        
        # Visit AST to find calls
        visitor = CallVisitor(file_path, function_lines, self.codebase_graph)
        visitor.visit(tree)
        
        # Add calls to graph
        for callee_name, call_sites in visitor.calls.items():
            if callee_name not in self.call_graph.calls:
                self.call_graph.calls[callee_name] = []
            self.call_graph.calls[callee_name].extend(call_sites)
            
            # Build caller -> callee mapping
            for call_site in call_sites:
                caller_key = f"{call_site.caller_file}:{call_site.caller_function}" if call_site.caller_function else call_site.caller_file
                if caller_key not in self.call_graph.callees:
                    self.call_graph.callees[caller_key] = []
                if callee_name not in self.call_graph.callees[caller_key]:
                    self.call_graph.callees[caller_key].append(callee_name)
                
                # Build callee -> caller mapping
                if callee_name not in self.call_graph.callers:
                    self.call_graph.callers[callee_name] = []
                if caller_key not in self.call_graph.callers[callee_name]:
                    self.call_graph.callers[callee_name].append(caller_key)
    
    def find_callers(self, function_name: str, file_path: Optional[str] = None) -> List[CallSite]:
        """
        Find all callers of a function.
        
        Args:
            function_name: Name of the function.
            file_path: Optional file path to narrow search.
        
        Returns:
            List of call sites.
        """
        callers: List[CallSite] = []
        
        # Find exact matches
        if function_name in self.call_graph.calls:
            for call_site in self.call_graph.calls[function_name]:
                if file_path is None or call_site.callee_file == file_path:
                    callers.append(call_site)
        
        return callers
    
    def find_callees(self, function_name: str, file_path: Optional[str] = None) -> List[str]:
        """
        Find all functions called by a function.
        
        Args:
            function_name: Name of the function.
            file_path: Optional file path.
        
        Returns:
            List of callee function names.
        """
        caller_key = f"{file_path}:{function_name}" if file_path else function_name
        
        # Try exact match first
        if caller_key in self.call_graph.callees:
            return self.call_graph.callees[caller_key]
        
        # Try partial match
        for key, callees in self.call_graph.callees.items():
            if key.endswith(f":{function_name}") or key == function_name:
                return callees
        
        return []
    
    def find_call_chain(self, start_function: str, end_function: str, max_depth: int = 10) -> List[str]:
        """
        Find call chain between two functions.
        
        Args:
            start_function: Starting function name.
            end_function: Ending function name.
            max_depth: Maximum depth to search.
        
        Returns:
            List of functions in the call chain.
        """
        from collections import deque
        
        queue = deque([(start_function, [start_function])])
        visited = {start_function}
        
        while queue and len(queue[0][1]) <= max_depth:
            current, path = queue.popleft()
            
            if current == end_function:
                return path
            
            # Find callees
            callees = self.find_callees(current)
            for callee in callees:
                if callee not in visited:
                    visited.add(callee)
                    queue.append((callee, path + [callee]))
        
        return []  # No chain found
    
    def find_dead_code(self) -> List[str]:
        """
        Find potentially dead code (functions that are never called).
        
        Returns:
            List of function names that appear to be unused.
        """
        dead_code: List[str] = []
        
        # Get all functions
        all_functions: Set[str] = set()
        for callees in self.call_graph.callees.values():
            all_functions.update(callees)
        
        # Find functions that are never called
        for function_name in all_functions:
            if function_name not in self.call_graph.callers:
                # Function is defined but never called (potential dead code)
                dead_code.append(function_name)
        
        return dead_code


class CallVisitor(ast.NodeVisitor):
    """AST visitor that finds function calls."""
    
    def __init__(self, file_path: str, function_lines: Dict[int, str], codebase_graph: Optional[CodebaseGraph] = None):
        self.file_path = file_path
        self.function_lines = function_lines
        self.codebase_graph = codebase_graph
        self.calls: Dict[str, List[CallSite]] = defaultdict(list)
        self.current_function: Optional[str] = None
        self.current_line = 0
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track current function."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track current async function."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node: ast.Call) -> None:
        """Find function calls."""
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr
        else:
            self.generic_visit(node)
            return
        
        # Resolve callee file if possible
        callee_file = None
        if self.codebase_graph:
            func_info = self.codebase_graph.get_function_info(callee_name)
            if func_info:
                callee_file = func_info.file_path
        
        call_site = CallSite(
            caller_file=self.file_path,
            caller_line=node.lineno,
            caller_function=self.current_function,
            callee_name=callee_name,
            callee_file=callee_file,
        )
        
        self.calls[callee_name].append(call_site)
        self.generic_visit(node)


def build_call_graph(repo_root: Path, codebase_graph: Optional[CodebaseGraph] = None) -> CallGraphBuilder:
    """
    Build call graph for a repository.
    
    Args:
        repo_root: Root directory of the repository.
        codebase_graph: Optional codebase graph for better symbol resolution.
    
    Returns:
        CallGraphBuilder object.
    """
    return CallGraphBuilder(repo_root, codebase_graph)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    
    # Build codebase graph first
    from memory.knowledge_graph import build_codebase_graph
    codebase_graph = build_codebase_graph(repo_root)
    
    # Build call graph
    call_graph_builder = build_call_graph(repo_root, codebase_graph)
    
    print(f"Call Graph for {repo_root}")
    print(f"Total calls tracked: {sum(len(sites) for sites in call_graph_builder.call_graph.calls.values())}")
    print()
    
    # Show example
    if call_graph_builder.call_graph.calls:
        func_name = list(call_graph_builder.call_graph.calls.keys())[0]
        call_sites = call_graph_builder.call_graph.calls[func_name]
        print(f"Function '{func_name}' is called {len(call_sites)} time(s):")
        for site in call_sites[:3]:
            print(f"  {site.caller_file}:{site.caller_line} (in {site.caller_function or 'module'})")
