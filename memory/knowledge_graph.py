"""Codebase knowledge graph - Claude Code level understanding."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict

try:
    from .symbols import extract_symbols, Symbol, FileSymbols
    from .dependencies import build_dependency_graph, DependencyGraph
    from .index import scan_repo
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from memory.symbols import extract_symbols, Symbol, FileSymbols
    from memory.dependencies import build_dependency_graph, DependencyGraph
    from memory.index import scan_repo


@dataclass
class ModuleInfo:
    """Information about a module/file."""
    
    path: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    source_file: Optional[str] = None  # If this is a test file


@dataclass
class FunctionInfo:
    """Information about a function."""
    
    name: str
    file_path: str
    line: int
    signature: str
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    callers: List[str] = field(default_factory=list)  # Functions that call this
    callees: List[str] = field(default_factory=list)  # Functions this calls


@dataclass
class ClassInfo:
    """Information about a class."""
    
    name: str
    file_path: str
    line: int
    docstring: Optional[str] = None
    parent_classes: List[str] = field(default_factory=list)  # Inheritance
    methods: List[str] = field(default_factory=list)  # Method names
    subclasses: List[str] = field(default_factory=list)  # Subclasses


@dataclass
class CodeLocation:
    """Location of code in the codebase."""
    
    file_path: str
    line: int
    column: int = 0
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None  # 'function', 'class', 'method', etc.


class CodebaseGraph:
    """Complete knowledge graph of the codebase."""
    
    def __init__(self, repo_root: Path):
        """
        Initialize codebase graph.
        
        Args:
            repo_root: Root directory of the repository.
        """
        self.repo_root = Path(repo_root).resolve()
        self.modules: Dict[str, ModuleInfo] = {}
        self.functions: Dict[str, FunctionInfo] = {}  # Key: "file_path:function_name"
        self.classes: Dict[str, ClassInfo] = {}  # Key: "file_path:class_name"
        self.symbol_index: Dict[str, List[CodeLocation]] = {}  # Symbol name -> locations
        self.dependency_graph: Optional[DependencyGraph] = None
        self._build_graph()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to a dict for caching."""
        modules: Dict[str, Dict[str, Any]] = {}
        for path, mod in self.modules.items():
            mod_dict = asdict(mod)
            modules[path] = mod_dict

        functions = {k: asdict(v) for k, v in self.functions.items()}
        classes = {k: asdict(v) for k, v in self.classes.items()}
        symbol_index = {
            name: [asdict(loc) for loc in locs] for name, locs in self.symbol_index.items()
        }
        return {
            "modules": modules,
            "functions": functions,
            "classes": classes,
            "symbol_index": symbol_index,
        }

    @classmethod
    def from_dict(cls, repo_root: Path, data: Dict[str, Any]) -> "CodebaseGraph":
        """Rehydrate graph from cached dict."""
        obj = cls.__new__(cls)
        obj.repo_root = Path(repo_root).resolve()
        obj.dependency_graph = None

        modules: Dict[str, ModuleInfo] = {}
        for path, mod in data.get("modules", {}).items():
            symbols = [Symbol(**s) for s in mod.get("symbols", [])]
            modules[path] = ModuleInfo(
                path=mod.get("path", path),
                symbols=symbols,
                imports=mod.get("imports", []),
                imported_by=mod.get("imported_by", []),
                test_files=mod.get("test_files", []),
                source_file=mod.get("source_file"),
            )
        obj.modules = modules

        obj.functions = {
            k: FunctionInfo(**v) for k, v in data.get("functions", {}).items()
        }
        obj.classes = {
            k: ClassInfo(**v) for k, v in data.get("classes", {}).items()
        }
        obj.symbol_index = {
            name: [CodeLocation(**loc) for loc in locs]
            for name, locs in data.get("symbol_index", {}).items()
        }
        return obj
    
    def _build_graph(self) -> None:
        """Build the complete knowledge graph."""
        # Build dependency graph first
        try:
            self.dependency_graph = build_dependency_graph(self.repo_root)
        except Exception:
            pass
        
        # Scan all Python files
        repo_files = scan_repo(str(self.repo_root))
        python_files = [f for f in repo_files if f.endswith(".py")]
        
        # Extract symbols from all files
        for file_path in python_files:
            self._process_file(file_path)
        
        # Build relationships (call graphs, inheritance, etc.)
        self._build_relationships()
    
    def _process_file(self, file_path: str) -> None:
        """Process a file and extract information."""
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return
        
        # Extract symbols
        file_symbols = extract_symbols(full_path, use_cache=True)
        
        # Create module info
        module_info = ModuleInfo(path=file_path)
        module_info.symbols = file_symbols.symbols
        module_info.imports = file_symbols.imports
        
        # Get dependencies
        if self.dependency_graph:
            deps = self.dependency_graph.get_dependencies(file_path)
            module_info.imported_by = deps.imported_by
            module_info.test_files = deps.test_files
            module_info.source_file = deps.source_file
        
        self.modules[file_path] = module_info
        
        # Index functions and classes
        for symbol in file_symbols.symbols:
            location = CodeLocation(
                file_path=file_path,
                line=symbol.line,
                column=symbol.column,
                symbol_name=symbol.name,
                symbol_type=symbol.kind,
            )
            
            # Add to symbol index
            if symbol.name not in self.symbol_index:
                self.symbol_index[symbol.name] = []
            self.symbol_index[symbol.name].append(location)
            
            # Create function/class info
            if symbol.kind in ("function", "method"):
                key = f"{file_path}:{symbol.name}"
                self.functions[key] = FunctionInfo(
                    name=symbol.name,
                    file_path=file_path,
                    line=symbol.line,
                    signature=symbol.signature or "()",
                    docstring=symbol.docstring,
                    parent_class=symbol.parent,
                )
            elif symbol.kind == "class":
                key = f"{file_path}:{symbol.name}"
                self.classes[key] = ClassInfo(
                    name=symbol.name,
                    file_path=file_path,
                    line=symbol.line,
                    docstring=symbol.docstring,
                )
    
    def _build_relationships(self) -> None:
        """Build relationships between symbols."""
        # Build inheritance relationships
        for file_path, module_info in self.modules.items():
            for symbol in module_info.symbols:
                if symbol.kind == "class":
                    key = f"{file_path}:{symbol.name}"
                    if key in self.classes:
                        # Extract base classes from AST (simplified)
                        # In a full implementation, we'd parse the AST more deeply
                        class_info = self.classes[key]
                        # Find methods in this class
                        for func_symbol in module_info.symbols:
                            if func_symbol.kind == "method" and func_symbol.parent == symbol.name:
                                class_info.methods.append(func_symbol.name)
        
        # Build call relationships (simplified - would need AST analysis)
        # For now, we'll use import relationships as a proxy
        if self.dependency_graph:
            for file_path, module_info in self.modules.items():
                for imported_file in module_info.imports:
                    if imported_file in self.modules:
                        imported_module = self.modules[imported_file]
                        # Functions in imported file are potential callees
                        for func_key, func_info in self.functions.items():
                            if func_info.file_path == imported_file:
                                # This is a potential callee (simplified)
                                pass
    
    def find_related_code(self, symbol_name: str) -> List[CodeLocation]:
        """
        Find all code related to a symbol.
        
        Args:
            symbol_name: Name of the symbol.
        
        Returns:
            List of related code locations.
        """
        related: List[CodeLocation] = []
        
        # Find symbol definitions
        if symbol_name in self.symbol_index:
            related.extend(self.symbol_index[symbol_name])
        
        # Find files that import this symbol
        for file_path, module_info in self.modules.items():
            if symbol_name in module_info.imports or symbol_name in [s.name for s in module_info.symbols]:
                # Add file location
                related.append(CodeLocation(
                    file_path=file_path,
                    line=0,
                    symbol_name=symbol_name,
                ))
        
        return related
    
    def find_callers(self, function_name: str, file_path: Optional[str] = None) -> List[CodeLocation]:
        """
        Find all callers of a function.
        
        Args:
            function_name: Name of the function.
            file_path: Optional file path to narrow search.
        
        Returns:
            List of call sites.
        """
        callers: List[CodeLocation] = []
        
        # Find function definition
        func_key = None
        if file_path:
            func_key = f"{file_path}:{function_name}"
        else:
            # Find first matching function
            for key, func_info in self.functions.items():
                if func_info.name == function_name:
                    func_key = key
                    break
        
        if not func_key or func_key not in self.functions:
            return callers
        
        func_info = self.functions[func_key]
        
        # Find files that import the function's module
        func_file = func_info.file_path
        for file_path, module_info in self.modules.items():
            if func_file in module_info.imports:
                # This file potentially calls the function
                # In a full implementation, we'd analyze AST to find actual call sites
                callers.append(CodeLocation(
                    file_path=file_path,
                    line=0,
                    symbol_name=function_name,
                    symbol_type="caller",
                ))
        
        return callers
    
    def find_callees(self, function_name: str, file_path: Optional[str] = None) -> List[CodeLocation]:
        """
        Find all functions called by a function.
        
        Args:
            function_name: Name of the function.
            file_path: Optional file path.
        
        Returns:
            List of called functions.
        """
        callees: List[CodeLocation] = []
        
        # Find function
        func_key = None
        if file_path:
            func_key = f"{file_path}:{function_name}"
        else:
            for key, func_info in self.functions.items():
                if func_info.name == function_name:
                    func_key = key
                    break
        
        if not func_key or func_key not in self.functions:
            return callees
        
        func_info = self.functions[func_key]
        
        # Find imported modules (potential callees)
        if func_info.file_path in self.modules:
            module_info = self.modules[func_info.file_path]
            for imported_file in module_info.imports:
                if imported_file in self.modules:
                    imported_module = self.modules[imported_file]
                    # Functions in imported module are potential callees
                    for symbol in imported_module.symbols:
                        if symbol.kind in ("function", "method"):
                            callees.append(CodeLocation(
                                file_path=imported_file,
                                line=symbol.line,
                                symbol_name=symbol.name,
                                symbol_type="callee",
                            ))
        
        return callees
    
    def get_dependency_path(self, file1: str, file2: str) -> List[str]:
        """
        Find dependency path between two files.
        
        Args:
            file1: Source file path.
            file2: Target file path.
        
        Returns:
            List of files in the dependency path.
        """
        if not self.dependency_graph:
            return []
        
        # Use BFS to find path
        from collections import deque
        
        queue = deque([(file1, [file1])])
        visited = {file1}
        
        while queue:
            current, path = queue.popleft()
            
            if current == file2:
                return path
            
            deps = self.dependency_graph.get_dependencies(current)
            for imported_file in deps.imports:
                if imported_file not in visited:
                    visited.add(imported_file)
                    queue.append((imported_file, path + [imported_file]))
        
        return []  # No path found
    
    def find_all_usages(self, symbol_name: str) -> List[CodeLocation]:
        """
        Find all usages of a symbol.
        
        Args:
            symbol_name: Name of the symbol.
        
        Returns:
            List of all usage locations.
        """
        usages: List[CodeLocation] = []
        
        # Direct definitions
        if symbol_name in self.symbol_index:
            usages.extend(self.symbol_index[symbol_name])
        
        # Find imports
        for file_path, module_info in self.modules.items():
            if symbol_name in module_info.imports:
                usages.append(CodeLocation(
                    file_path=file_path,
                    line=0,
                    symbol_name=symbol_name,
                    symbol_type="import",
                ))
        
        return usages
    
    def get_module_info(self, file_path: str) -> Optional[ModuleInfo]:
        """Get information about a module."""
        return self.modules.get(file_path)
    
    def get_function_info(self, function_name: str, file_path: Optional[str] = None) -> Optional[FunctionInfo]:
        """Get information about a function."""
        if file_path:
            key = f"{file_path}:{function_name}"
            return self.functions.get(key)
        
        # Find first matching function
        for key, func_info in self.functions.items():
            if func_info.name == function_name:
                return func_info
        
        return None
    
    def get_class_info(self, class_name: str, file_path: Optional[str] = None) -> Optional[ClassInfo]:
        """Get information about a class."""
        if file_path:
            key = f"{file_path}:{class_name}"
            return self.classes.get(key)
        
        # Find first matching class
        for key, class_info in self.classes.items():
            if class_info.name == class_name:
                return class_info
        
        return None


def build_codebase_graph(repo_root: Path) -> CodebaseGraph:
    """
    Build a complete codebase knowledge graph.
    
    Args:
        repo_root: Root directory of the repository.
    
    Returns:
        CodebaseGraph object.
    """
    return CodebaseGraph(repo_root)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    graph = build_codebase_graph(repo_root)
    
    print(f"Codebase Graph for {repo_root}")
    print(f"Modules: {len(graph.modules)}")
    print(f"Functions: {len(graph.functions)}")
    print(f"Classes: {len(graph.classes)}")
    print()
    
    # Show example
    if graph.functions:
        func_key = list(graph.functions.keys())[0]
        func_info = graph.functions[func_key]
        print(f"Example function: {func_info.name}")
        print(f"  File: {func_info.file_path}")
        print(f"  Line: {func_info.line}")
        print(f"  Signature: {func_info.signature}")
