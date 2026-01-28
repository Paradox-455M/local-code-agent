"""File dependency graph and relationship analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional

try:
    from .symbols import extract_symbols, FileSymbols
    from .index import scan_repo
    from ..core.config import config
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from memory.symbols import extract_symbols, FileSymbols
    from memory.index import scan_repo
    from core.config import config


@dataclass
class FileDependencies:
    """Dependencies for a single file."""

    file_path: str
    imports: List[str] = field(default_factory=list)  # Files this file imports
    imported_by: List[str] = field(default_factory=list)  # Files that import this file
    test_files: List[str] = field(default_factory=list)  # Related test files
    source_file: Optional[str] = None  # Source file if this is a test file


class DependencyGraph:
    """Graph of file dependencies in a repository."""

    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root)
        self.dependencies: Dict[str, FileDependencies] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the dependency graph by analyzing all Python files."""
        repo_files = scan_repo(str(self.repo_root))
        python_files = [f for f in repo_files if f.endswith(".py")]

        # First pass: extract imports from all files
        for file_path in python_files:
            full_path = self.repo_root / file_path
            file_symbols = extract_symbols(full_path, use_cache=True)
            deps = FileDependencies(file_path=file_path)
            deps.imports = self._resolve_imports(file_symbols.imports, file_path, python_files)
            self.dependencies[file_path] = deps

        # Second pass: build reverse dependencies (imported_by)
        for file_path, deps in self.dependencies.items():
            for imported_file in deps.imports:
                if imported_file in self.dependencies:
                    self.dependencies[imported_file].imported_by.append(file_path)

        # Third pass: identify test files and their source files
        self._identify_test_files()

    def _resolve_imports(self, imports: List[str], current_file: str, all_files: List[str]) -> List[str]:
        """
        Resolve import module names to actual file paths.

        Args:
            imports: List of imported module names.
            current_file: Path of the file doing the importing.
            all_files: List of all Python files in the repo.

        Returns:
            List of file paths (relative to repo_root) that are imported.
        """
        resolved: List[str] = []
        current_dir = Path(current_file).parent

        for imp in imports:
            # Try to resolve to a file in the repo
            # Handle different import styles:
            # 1. Absolute imports: "package.module" -> "package/module.py"
            # 2. Relative imports: ".module" -> relative to current file
            # 3. Local imports: "module" -> same directory or package

            # Convert module name to potential file paths
            candidates = self._module_to_file_candidates(imp, current_dir, all_files)

            for candidate in candidates:
                if candidate in all_files:
                    if candidate not in resolved:
                        resolved.append(candidate)
                    break

        return resolved

    def _module_to_file_candidates(self, module_name: str, current_dir: Path, all_files: List[str]) -> List[str]:
        """Convert a module name to potential file path candidates."""
        candidates: List[str] = []

        # Remove leading dots (relative imports)
        module_name = module_name.lstrip(".")

        # Convert dots to path separators
        parts = module_name.split(".")
        base_path = "/".join(parts)

        # Common patterns:
        # 1. Direct file: "module" -> "module.py", "current_dir/module.py"
        # 2. Package: "package.module" -> "package/module.py"
        # 3. Subpackage: "package.sub.module" -> "package/sub/module.py"

        # Try as direct file in current directory
        if current_dir != Path("."):
            candidates.append(str(current_dir / f"{parts[-1]}.py"))

        # Try as package module
        candidates.append(f"{base_path}.py")

        # Try with __init__.py
        if len(parts) > 1:
            candidates.append(f"{'/'.join(parts[:-1])}/__init__.py")
            candidates.append(f"{'/'.join(parts[:-1])}/{parts[-1]}.py")

        # Try from repo root
        candidates.append(f"{base_path}.py")

        return candidates

    def _identify_test_files(self) -> None:
        """Identify test files and link them to source files."""
        test_patterns = ["test_", "_test", "tests/"]

        for file_path, deps in self.dependencies.items():
            path_lower = file_path.lower()
            is_test = any(pattern in path_lower for pattern in test_patterns)

            if is_test:
                # Try to find corresponding source file
                source_file = self._find_source_file(file_path)
                if source_file:
                    deps.source_file = source_file
                    # Add reverse link
                    if source_file in self.dependencies:
                        self.dependencies[source_file].test_files.append(file_path)

    def _find_source_file(self, test_file: str) -> Optional[str]:
        """Find the source file corresponding to a test file."""
        test_path = Path(test_file)
        test_name = test_path.stem

        # Remove test prefixes/suffixes
        source_name = test_name
        for prefix in ["test_", "tests_"]:
            if source_name.startswith(prefix):
                source_name = source_name[len(prefix) :]
        for suffix in ["_test", "_tests"]:
            if source_name.endswith(suffix):
                source_name = source_name[: -len(suffix)]

        # Try to find matching source file
        repo_files = scan_repo(str(self.repo_root))
        python_files = [f for f in repo_files if f.endswith(".py") and not any(p in f.lower() for p in ["test", "tests/"])]

        # Check same directory first
        test_dir = test_path.parent
        for candidate in python_files:
            candidate_path = Path(candidate)
            if candidate_path.parent == test_dir and candidate_path.stem == source_name:
                return candidate

        # Check if source_name matches any file
        for candidate in python_files:
            if Path(candidate).stem == source_name:
                return candidate

        return None

    def get_dependencies(self, file_path: str) -> FileDependencies:
        """Get dependencies for a specific file."""
        return self.dependencies.get(file_path, FileDependencies(file_path=file_path))

    def get_related_files(self, file_path: str, max_depth: int = 2) -> Set[str]:
        """
        Get files related to the given file through imports.

        Args:
            file_path: Path to the file.
            max_depth: Maximum depth to traverse (1 = direct imports only).

        Returns:
            Set of related file paths.
        """
        related: Set[str] = set()
        if file_path not in self.dependencies:
            return related

        def _traverse(current: str, depth: int) -> None:
            if depth > max_depth or current in related:
                return
            related.add(current)
            deps = self.dependencies.get(current)
            if deps:
                for imported in deps.imports:
                    _traverse(imported, depth + 1)
                for importer in deps.imported_by:
                    _traverse(importer, depth + 1)

        _traverse(file_path, 0)
        related.discard(file_path)  # Don't include the file itself
        return related

    def find_files_using_symbol(self, symbol_name: str) -> List[str]:
        """
        Find files that use (import or define) a symbol.

        Args:
            symbol_name: Name of the symbol to search for.

        Returns:
            List of file paths using the symbol.
        """
        from .symbols import find_files_with_symbol

        return find_files_with_symbol(symbol_name, self.repo_root, use_cache=True)


def build_dependency_graph(repo_root: str | Path) -> DependencyGraph:
    """
    Build a dependency graph for a repository.

    Args:
        repo_root: Root directory of the repository.

    Returns:
        DependencyGraph object.
    """
    return DependencyGraph(repo_root)


if __name__ == "__main__":
    # Demo: Build dependency graph
    import sys

    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."
    graph = build_dependency_graph(repo_root)

    print(f"Dependency Graph for {repo_root}")
    print(f"Total files: {len(graph.dependencies)}")
    print()

    # Show a few examples
    for file_path, deps in list(graph.dependencies.items())[:5]:
        print(f"File: {file_path}")
        if deps.imports:
            print(f"  Imports: {', '.join(deps.imports[:3])}")
        if deps.imported_by:
            print(f"  Imported by: {', '.join(deps.imported_by[:3])}")
        if deps.test_files:
            print(f"  Test files: {', '.join(deps.test_files)}")
        print()
