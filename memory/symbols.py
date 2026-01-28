"""AST-based symbol extraction and semantic code search."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional

try:
    from .index import scan_repo
    from ..core.config import config
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from memory.index import scan_repo
    from core.config import config


@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable, etc.)."""

    name: str
    kind: str  # 'function', 'class', 'variable', 'import', 'method'
    file_path: str
    line: int
    column: int = 0
    docstring: Optional[str] = None
    signature: Optional[str] = None  # For functions/methods
    parent: Optional[str] = None  # Parent class for methods


@dataclass
class FileSymbols:
    """Symbols extracted from a single file."""

    file_path: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)  # Imported module names
    imported_symbols: List[str] = field(default_factory=list)  # Imported symbol names
    parse_error: Optional[str] = None


# Cache for parsed ASTs and symbols
_symbol_cache: Dict[str, FileSymbols] = {}
_cache_timestamps: Dict[str, float] = {}


def extract_symbols(file_path: str | Path, use_cache: bool = True) -> FileSymbols:
    """
    Extract symbols from a Python file using AST parsing.

    Args:
        file_path: Path to the Python file (relative or absolute).
        use_cache: Whether to use cached results if available.

    Returns:
        FileSymbols object containing extracted symbols and imports.
    """
    file_path_str = str(file_path)
    file_path_obj = Path(file_path_str)

    # Check cache
    if use_cache and file_path_str in _symbol_cache:
        # Verify file hasn't changed
        try:
            if file_path_obj.exists():
                current_mtime = file_path_obj.stat().st_mtime
                cached_mtime = _cache_timestamps.get(file_path_str, 0)
                if current_mtime <= cached_mtime:
                    return _symbol_cache[file_path_str]
        except OSError:
            pass

    result = FileSymbols(file_path=file_path_str)

    if not file_path_obj.exists():
        result.parse_error = "File does not exist"
        return result

    if not file_path_obj.suffix == ".py":
        result.parse_error = "Not a Python file"
        return result

    try:
        source = file_path_obj.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=file_path_str)
    except SyntaxError as e:
        result.parse_error = f"Syntax error: {e}"
        return result
    except Exception as e:
        result.parse_error = f"Parse error: {e}"
        return result

    # Extract symbols
    visitor = SymbolExtractor(file_path_str)
    visitor.visit(tree)
    result.symbols = visitor.symbols
    result.imports = visitor.imports
    result.imported_symbols = visitor.imported_symbols

    # Update cache
    if use_cache:
        _symbol_cache[file_path_str] = result
        try:
            _cache_timestamps[file_path_str] = file_path_obj.stat().st_mtime
        except OSError:
            pass

    return result


class SymbolExtractor(ast.NodeVisitor):
    """AST visitor that extracts symbols from Python code."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: List[Symbol] = []
        self.imports: List[str] = []
        self.imported_symbols: List[str] = []
        self.current_class: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definitions."""
        kind = "method" if self.current_class else "function"
        parent = self.current_class

        # Extract signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"({', '.join(args)})"

        symbol = Symbol(
            name=node.name,
            kind=kind,
            file_path=self.file_path,
            line=node.lineno,
            column=node.col_offset,
            docstring=ast.get_docstring(node),
            signature=signature,
            parent=parent,
        )
        self.symbols.append(symbol)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract async function definitions."""
        kind = "method" if self.current_class else "function"
        parent = self.current_class

        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"({', '.join(args)})"

        symbol = Symbol(
            name=node.name,
            kind=kind,
            file_path=self.file_path,
            line=node.lineno,
            column=node.col_offset,
            docstring=ast.get_docstring(node),
            signature=signature,
            parent=parent,
        )
        self.symbols.append(symbol)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definitions."""
        old_class = self.current_class
        self.current_class = node.name

        symbol = Symbol(
            name=node.name,
            kind="class",
            file_path=self.file_path,
            line=node.lineno,
            column=node.col_offset,
            docstring=ast.get_docstring(node),
            parent=None,
        )
        self.symbols.append(symbol)
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Import(self, node: ast.Import) -> None:
        """Extract import statements."""
        for alias in node.names:
            module_name = alias.name
            self.imports.append(module_name)
            if alias.asname:
                self.imported_symbols.append(alias.asname)
            else:
                # Extract top-level name from module
                top_level = module_name.split(".")[0]
                self.imported_symbols.append(top_level)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from-import statements."""
        if node.module:
            self.imports.append(node.module)
        for alias in node.names:
            symbol_name = alias.asname if alias.asname else alias.name
            self.imported_symbols.append(symbol_name)
        self.generic_visit(node)


def find_symbols_by_name(symbol_name: str, repo_root: str | Path, use_cache: bool = True) -> List[Symbol]:
    """
    Find all symbols matching a name across the repository.

    Args:
        symbol_name: Name of the symbol to search for.
        repo_root: Root directory of the repository.
        use_cache: Whether to use cached symbol extraction.

    Returns:
        List of matching symbols.
    """
    repo_files = scan_repo(str(repo_root))
    matching_symbols: List[Symbol] = []
    symbol_name_lower = symbol_name.lower()

    for file_path in repo_files:
        if not file_path.endswith(".py"):
            continue

        full_path = Path(repo_root) / file_path
        file_symbols = extract_symbols(full_path, use_cache=use_cache)

        for symbol in file_symbols.symbols:
            if symbol.name.lower() == symbol_name_lower:
                matching_symbols.append(symbol)

        # Also check imported symbols
        for imported in file_symbols.imported_symbols:
            if imported.lower() == symbol_name_lower:
                # Create a pseudo-symbol for imported items
                matching_symbols.append(
                    Symbol(
                        name=imported,
                        kind="import",
                        file_path=file_path,
                        line=0,
                    )
                )

    return matching_symbols


def find_files_with_symbol(symbol_name: str, repo_root: str | Path, use_cache: bool = True) -> List[str]:
    """
    Find files that contain or import a symbol.

    Args:
        symbol_name: Name of the symbol to search for.
        repo_root: Root directory of the repository.
        use_cache: Whether to use cached symbol extraction.

    Returns:
        List of file paths (relative to repo_root) containing the symbol.
    """
    symbols = find_symbols_by_name(symbol_name, repo_root, use_cache)
    files = list(set(s.file_path for s in symbols))
    return files


def score_file_by_symbols(file_path: str, symbol_names: List[str], repo_root: str | Path, use_cache: bool = True) -> float:
    """
    Score a file based on how many symbols it contains or imports.

    Args:
        file_path: Path to the file (relative to repo_root).
        symbol_names: List of symbol names to search for.
        repo_root: Root directory of the repository.
        use_cache: Whether to use cached symbol extraction.

    Returns:
        Score (higher = more relevant).
    """
    if not symbol_names:
        return 0.0

    full_path = Path(repo_root) / file_path
    if not full_path.exists() or not full_path.suffix == ".py":
        return 0.0

    file_symbols = extract_symbols(full_path, use_cache=use_cache)
    score = 0.0

    symbol_names_lower = [s.lower() for s in symbol_names]

    # Score based on defined symbols
    for symbol in file_symbols.symbols:
        if symbol.name.lower() in symbol_names_lower:
            # Functions and classes are more important
            if symbol.kind in ("function", "class"):
                score += 3.0
            elif symbol.kind == "method":
                score += 2.0
            else:
                score += 1.0

    # Score based on imported symbols
    for imported in file_symbols.imported_symbols:
        if imported.lower() in symbol_names_lower:
            score += 1.5  # Imports are relevant but less than definitions

    # Score based on imports (module names)
    for imp in file_symbols.imports:
        for sym_name in symbol_names:
            if sym_name.lower() in imp.lower():
                score += 0.5

    return score


def clear_cache() -> None:
    """Clear the symbol extraction cache."""
    global _symbol_cache, _cache_timestamps
    _symbol_cache.clear()
    _cache_timestamps.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about the cache."""
    return {
        "cached_files": len(_symbol_cache),
        "cache_entries": sum(len(fs.symbols) for fs in _symbol_cache.values()),
    }


if __name__ == "__main__":
    # Demo: Extract symbols from a file
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        symbols = extract_symbols(test_file)
        print(f"File: {symbols.file_path}")
        print(f"Symbols: {len(symbols.symbols)}")
        for sym in symbols.symbols[:10]:  # Show first 10
            print(f"  {sym.kind}: {sym.name} (line {sym.line})")
        print(f"Imports: {len(symbols.imports)}")
        for imp in symbols.imports[:5]:  # Show first 5
            print(f"  {imp}")
