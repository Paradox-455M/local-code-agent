"""Intelligent context builder for code generation - Claude Code level."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

try:
    from memory.symbols import extract_symbols, Symbol, find_files_with_symbol
    from memory.dependencies import build_dependency_graph, DependencyGraph
    from memory.index import scan_repo
    from memory.knowledge_graph import build_codebase_graph, CodebaseGraph
    from memory.call_graph import build_call_graph, CallGraphBuilder
    from memory.semantic_search import create_semantic_search, SemanticCodeSearch
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from memory.symbols import extract_symbols, Symbol, find_files_with_symbol
    from memory.dependencies import build_dependency_graph, DependencyGraph
    from memory.index import scan_repo
    try:
        from memory.knowledge_graph import build_codebase_graph, CodebaseGraph
        from memory.call_graph import build_call_graph, CallGraphBuilder
        from memory.semantic_search import create_semantic_search, SemanticCodeSearch
    except ImportError:
        CodebaseGraph = None
        CallGraphBuilder = None
        SemanticCodeSearch = None
        build_codebase_graph = None
        build_call_graph = None
        create_semantic_search = None


@dataclass
class CodeSnippet:
    """A code snippet with metadata."""
    
    path: str
    content: str
    start_line: int = 0
    end_line: int = 0
    relevance_score: float = 0.0
    snippet_type: str = "code"  # 'code', 'test', 'doc', 'import', 'related'
    symbols: List[str] = field(default_factory=list)


@dataclass
class Context:
    """Intelligent context for code generation."""
    
    snippets: List[CodeSnippet] = field(default_factory=list)
    primary_files: Set[str] = field(default_factory=set)
    related_files: Set[str] = field(default_factory=set)
    test_files: Set[str] = field(default_factory=set)
    total_bytes: int = 0
    max_bytes: int = 40000  # Default max context size
    
    def add_snippet(self, snippet: CodeSnippet) -> None:
        """Add a snippet to context."""
        self.snippets.append(snippet)
        self.total_bytes += len(snippet.content.encode("utf-8"))
    
    def prioritize(self) -> None:
        """Prioritize snippets by relevance score."""
        self.snippets.sort(key=lambda s: -s.relevance_score)
    
    def prune(self, max_bytes: Optional[int] = None) -> None:
        """Prune context to fit within byte limit."""
        max_bytes = max_bytes or self.max_bytes
        pruned: List[CodeSnippet] = []
        current_bytes = 0
        
        for snippet in self.snippets:
            snippet_bytes = len(snippet.content.encode("utf-8"))
            if current_bytes + snippet_bytes <= max_bytes:
                pruned.append(snippet)
                current_bytes += snippet_bytes
            else:
                # Try to fit partial snippet
                remaining = max_bytes - current_bytes
                if remaining > 100:  # Only if meaningful space left
                    partial = snippet.content[:remaining]
                    pruned.append(CodeSnippet(
                        path=snippet.path,
                        content=partial + "\n... (truncated)",
                        start_line=snippet.start_line,
                        end_line=snippet.end_line,
                        relevance_score=snippet.relevance_score,
                        snippet_type=snippet.snippet_type,
                        symbols=snippet.symbols
                    ))
                break
        
        self.snippets = pruned
        self.total_bytes = current_bytes
    
    def to_dict_list(self) -> List[Dict[str, str]]:
        """Convert to list of dicts for executor compatibility."""
        return [
            {
                "path": snippet.path,
                "snippet": snippet.content,
                "type": snippet.snippet_type,
                "relevance": snippet.relevance_score,
            }
            for snippet in self.snippets
        ]


class IntelligentContextBuilder:
    """Builds intelligent context like Claude Code."""
    
    def __init__(self, repo_root: Path, use_knowledge_graph: bool = True):
        """
        Initialize context builder.
        
        Args:
            repo_root: Root directory of the repository.
            use_knowledge_graph: Whether to use knowledge graph for better context.
        """
        self.repo_root = Path(repo_root).resolve()
        self.dependency_graph: Optional[DependencyGraph] = None
        self.knowledge_graph: Optional[CodebaseGraph] = None
        self.call_graph: Optional[CallGraphBuilder] = None
        self.semantic_search: Optional[SemanticCodeSearch] = None
        self._symbol_cache: Dict[str, List[Symbol]] = {}
        self.use_knowledge_graph = use_knowledge_graph and CodebaseGraph is not None
        
        if self.use_knowledge_graph:
            self._initialize_knowledge_graph()
    
    def _initialize_knowledge_graph(self) -> None:
        """Initialize knowledge graph and related structures."""
        try:
            if build_codebase_graph:
                self.knowledge_graph = build_codebase_graph(self.repo_root)
            if build_call_graph and self.knowledge_graph:
                self.call_graph = build_call_graph(self.repo_root, self.knowledge_graph)
            if create_semantic_search:
                self.semantic_search = create_semantic_search(self.repo_root)
        except Exception:
            # If knowledge graph building fails, continue without it
            self.use_knowledge_graph = False
    
    def _ensure_dependency_graph(self) -> None:
        """Ensure dependency graph is built."""
        if self.dependency_graph is None:
            try:
                self.dependency_graph = build_dependency_graph(self.repo_root)
            except Exception:
                # If graph building fails, continue without it
                pass
    
    def build_context(
        self,
        task: str,
        files: List[str],
        symbols: Optional[List[str]] = None,
        max_files: int = 10,
        max_bytes: int = 40000,
        include_tests: bool = True,
        include_related: bool = True,
    ) -> Context:
        """
        Build intelligent context for code generation.
        
        Args:
            task: The task description.
            files: Primary files to include.
            symbols: Optional symbols to prioritize.
            max_files: Maximum number of files to include.
            max_bytes: Maximum total bytes for context.
            include_tests: Whether to include test files.
            include_related: Whether to include related files.
        
        Returns:
            Context object with prioritized snippets.
        """
        context = Context(max_bytes=max_bytes)
        context.primary_files = set(files)
        
        # Build dependency graph if needed
        if include_related:
            self._ensure_dependency_graph()
        
        # Extract symbols from task if not provided
        if not symbols:
            symbols = self._extract_symbols_from_task(task)
        
        # Find related files
        related_files = self._find_related_files(files, include_tests, include_related)
        context.related_files = related_files
        
        # Find test files
        if include_tests:
            test_files = self._find_test_files(files)
            context.test_files = test_files
            related_files.update(test_files)
        
        # Score and select files
        all_files = list(set(files) | related_files)
        scored_files = self._score_files(all_files, files, symbols)
        
        # Select top files
        selected_files = [f for f, _ in scored_files[:max_files]]
        
        # Build snippets from selected files
        for file_path in selected_files:
            snippets = self._build_snippets_for_file(
                file_path,
                symbols,
                is_primary=file_path in files,
                is_test=file_path in context.test_files,
            )
            for snippet in snippets:
                context.add_snippet(snippet)
        
        # Prioritize and prune
        context.prioritize()
        context.prune(max_bytes)
        
        return context
    
    def _extract_symbols_from_task(self, task: str) -> List[str]:
        """Extract potential symbols from task description."""
        # Simple keyword extraction (can be enhanced with NLP)
        import re
        # Look for function/class names (capitalized words, words followed by parentheses)
        symbols = []
        
        # Function calls: function_name(...)
        func_pattern = r'\b([a-z_][a-z0-9_]*)\s*\('
        symbols.extend(re.findall(func_pattern, task))
        
        # Class names: ClassName
        class_pattern = r'\b([A-Z][a-zA-Z0-9_]*)\b'
        symbols.extend(re.findall(class_pattern, task))
        
        # Quoted names: "function_name" or 'function_name'
        quoted_pattern = r'["\']([a-z_][a-z0-9_]*)["\']'
        symbols.extend(re.findall(quoted_pattern, task))
        
        # Remove duplicates and common words
        common_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by'}
        symbols = [s for s in set(symbols) if s.lower() not in common_words and len(s) > 2]
        
        return symbols
    
    def _find_related_files(
        self,
        files: List[str],
        include_tests: bool,
        include_related: bool,
    ) -> Set[str]:
        """Find files related to the primary files."""
        related: Set[str] = set()
        
        if not include_related:
            return related
        
        # Use knowledge graph if available
        if self.use_knowledge_graph and self.knowledge_graph:
            for file_path in files:
                if file_path.endswith(".py"):
                    module_info = self.knowledge_graph.get_module_info(file_path)
                    if module_info:
                        # Add imports
                        related.update(module_info.imports)
                        # Add files that import this file
                        related.update(module_info.imported_by)
                        # Add test files
                        if include_tests:
                            related.update(module_info.test_files)
        elif self.dependency_graph:
            # Fallback to dependency graph
            for file_path in files:
                if file_path.endswith(".py"):
                    try:
                        deps = self.dependency_graph.get_dependencies(file_path)
                        related.update(deps.imports)
                        related.update(deps.imported_by)
                        related_files = self.dependency_graph.get_related_files(file_path, max_depth=1)
                        related.update(related_files)
                    except Exception:
                        continue
        
        return related
    
    def _find_test_files(self, files: List[str]) -> Set[str]:
        """Find test files corresponding to source files."""
        test_files: Set[str] = set()
        
        if self.use_knowledge_graph and self.knowledge_graph:
            for file_path in files:
                if file_path.endswith(".py"):
                    module_info = self.knowledge_graph.get_module_info(file_path)
                    if module_info:
                        test_files.update(module_info.test_files)
        elif self.dependency_graph:
            for file_path in files:
                if file_path.endswith(".py"):
                    try:
                        deps = self.dependency_graph.get_dependencies(file_path)
                        test_files.update(deps.test_files)
                    except Exception:
                        continue
        
        return test_files
    
    def _score_files(
        self,
        files: List[str],
        primary_files: List[str],
        symbols: List[str],
    ) -> List[Tuple[str, float]]:
        """Score files by relevance."""
        scored: List[Tuple[str, float]] = []
        
        for file_path in files:
            score = 0.0
            
            # Base score: primary files are more important
            if file_path in primary_files:
                score += 100.0
            else:
                score += 10.0
            
            # Symbol-based scoring
            if symbols and file_path.endswith(".py"):
                try:
                    from memory.symbols import score_file_by_symbols
                    symbol_score = score_file_by_symbols(
                        file_path,
                        symbols,
                        self.repo_root,
                        use_cache=True
                    )
                    score += symbol_score * 5.0
                except Exception:
                    pass
            
            # Path-based scoring (shorter paths preferred)
            path_depth = len(Path(file_path).parts)
            score -= path_depth * 0.5
            
            scored.append((file_path, score))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: -x[1])
        
        return scored
    
    def _build_snippets_for_file(
        self,
        file_path: str,
        symbols: List[str],
        is_primary: bool = False,
        is_test: bool = False,
    ) -> List[CodeSnippet]:
        """Build relevant snippets from a file."""
        snippets: List[CodeSnippet] = []
        
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return snippets
        
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return snippets
        
        # Extract symbols from file
        file_symbols = extract_symbols(full_path, use_cache=True)
        
        # If file is small, include entire file
        if len(content.encode("utf-8")) < 2000:
            snippet = CodeSnippet(
                path=file_path,
                content=content,
                relevance_score=100.0 if is_primary else 50.0,
                snippet_type="test" if is_test else "code",
                symbols=[s.name for s in file_symbols.symbols],
            )
            snippets.append(snippet)
            return snippets
        
        # For larger files, extract relevant snippets
        lines = content.splitlines()
        
        # Find lines containing symbols
        symbol_lines: Set[int] = set()
        for symbol in file_symbols.symbols:
            symbol_lines.add(symbol.line - 1)  # Convert to 0-indexed
        
        # Also find lines containing symbol names from task
        if symbols:
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(sym.lower() in line_lower for sym in symbols):
                    symbol_lines.add(i)
        
        # Build snippets around symbol lines
        if symbol_lines:
            # Group nearby lines
            line_groups = self._group_nearby_lines(sorted(symbol_lines), max_gap=10)
            
            for group in line_groups:
                start = max(0, group[0] - 5)  # 5 lines before
                end = min(len(lines), group[-1] + 15)  # 15 lines after
                
                snippet_content = "\n".join(lines[start:end])
                
                # Calculate relevance score
                relevance = self._calculate_relevance(
                    snippet_content,
                    symbols,
                    is_primary,
                    is_test,
                )
                
                snippet = CodeSnippet(
                    path=file_path,
                    content=snippet_content,
                    start_line=start + 1,
                    end_line=end,
                    relevance_score=relevance,
                    snippet_type="test" if is_test else "code",
                    symbols=[s.name for s in file_symbols.symbols if start <= s.line - 1 < end],
                )
                snippets.append(snippet)
        else:
            # No symbols found, include beginning of file
            snippet_content = "\n".join(lines[:50])
            snippet = CodeSnippet(
                path=file_path,
                content=snippet_content,
                start_line=1,
                end_line=50,
                relevance_score=30.0 if is_primary else 10.0,
                snippet_type="test" if is_test else "code",
            )
            snippets.append(snippet)
        
        return snippets
    
    def _group_nearby_lines(self, lines: List[int], max_gap: int = 10) -> List[List[int]]:
        """Group nearby line numbers together."""
        if not lines:
            return []
        
        groups: List[List[int]] = []
        current_group = [lines[0]]
        
        for line in lines[1:]:
            if line - current_group[-1] <= max_gap:
                current_group.append(line)
            else:
                groups.append(current_group)
                current_group = [line]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _calculate_relevance(
        self,
        content: str,
        symbols: List[str],
        is_primary: bool,
        is_test: bool,
    ) -> float:
        """Calculate relevance score for a snippet."""
        score = 50.0 if is_primary else 20.0
        
        if is_test:
            score += 10.0
        
        # Boost score for symbol matches
        content_lower = content.lower()
        for symbol in symbols:
            if symbol.lower() in content_lower:
                score += 15.0
        
        return score
    
    def expand_context(self, context: Context, task: str) -> Context:
        """
        Expand context with additional related code.
        
        Args:
            context: Existing context.
            task: Task description.
        
        Returns:
            Expanded context.
        """
        # Find additional related files based on symbols in context
        all_symbols: Set[str] = set()
        for snippet in context.snippets:
            all_symbols.update(snippet.symbols)
        
        # Use knowledge graph if available
        if self.use_knowledge_graph and self.knowledge_graph:
            additional_files: Set[str] = set()
            
            # Find related code using knowledge graph
            for symbol in all_symbols:
                related_locations = self.knowledge_graph.find_related_code(symbol)
                for location in related_locations:
                    if location.file_path not in context.primary_files:
                        additional_files.add(location.file_path)
            
            # Use semantic search if available
            if self.semantic_search and task:
                semantic_matches = self.semantic_search.search(task, top_k=5)
                for match in semantic_matches:
                    if match.file_path not in context.primary_files:
                        additional_files.add(match.file_path)
        elif self.dependency_graph:
            # Fallback to dependency graph
            additional_files: Set[str] = set()
            for symbol in all_symbols:
                files = self.dependency_graph.find_files_using_symbol(symbol)
                additional_files.update(files)
        else:
            additional_files = set()
        
        # Add new snippets from additional files
        for file_path in additional_files:
            if file_path not in context.primary_files and file_path not in context.related_files:
                snippets = self._build_snippets_for_file(file_path, list(all_symbols))
                for snippet in snippets:
                    if snippet.relevance_score > 20.0:  # Only add relevant snippets
                        context.add_snippet(snippet)
        
        # Re-prioritize and prune
        context.prioritize()
        context.prune()
        
        return context


def build_intelligent_context(
    repo_root: Path,
    task: str,
    files: List[str],
    symbols: Optional[List[str]] = None,
    max_files: int = 10,
    max_bytes: int = 40000,
    include_tests: bool = True,
    include_related: bool = True,
) -> Context:
    """
    Build intelligent context for code generation.
    
    Args:
        repo_root: Repository root directory.
        task: Task description.
        files: Primary files to include.
        symbols: Optional symbols to prioritize.
        max_files: Maximum number of files.
        max_bytes: Maximum context size in bytes.
        include_tests: Whether to include test files.
        include_related: Whether to include related files.
    
    Returns:
        Context object.
    """
    builder = IntelligentContextBuilder(repo_root)
    return builder.build_context(
        task=task,
        files=files,
        symbols=symbols,
        max_files=max_files,
        max_bytes=max_bytes,
        include_tests=include_tests,
        include_related=include_related,
    )


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    task = sys.argv[2] if len(sys.argv) > 2 else "fix bug"
    files = sys.argv[3].split(",") if len(sys.argv) > 3 else []
    
    builder = IntelligentContextBuilder(repo_root)
    context = builder.build_context(task, files)
    
    print(f"Context built: {len(context.snippets)} snippets")
    print(f"Total bytes: {context.total_bytes}")
    print(f"Primary files: {context.primary_files}")
    print(f"Related files: {len(context.related_files)}")
    print(f"Test files: {context.test_files}")
    print("\nSnippets:")
    for snippet in context.snippets[:5]:
        print(f"  {snippet.path} (score: {snippet.relevance_score:.1f}, type: {snippet.snippet_type})")
