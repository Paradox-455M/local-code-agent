"""Semantic code search using embeddings - Claude Code level."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import json

try:
    from .symbols import extract_symbols, Symbol
    from .index import scan_repo
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from memory.symbols import extract_symbols, Symbol
    from memory.index import scan_repo


@dataclass
class CodeChunk:
    """A chunk of code with embedding."""
    
    file_path: str
    content: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class CodeMatch:
    """A code match from semantic search."""
    
    file_path: str
    content: str
    similarity_score: float
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    match_type: str = "semantic"  # 'semantic', 'exact', 'symbol'


class SemanticCodeSearch:
    """Semantic code search using embeddings."""
    
    def __init__(self, repo_root: Path, use_local_model: bool = True):
        """
        Initialize semantic code search.
        
        Args:
            repo_root: Repository root directory.
            use_local_model: Whether to use local embedding model.
        """
        self.repo_root = Path(repo_root).resolve()
        self.use_local_model = use_local_model
        self.chunks: List[CodeChunk] = []
        self.embeddings_cache: Dict[str, List[float]] = {}
        self._embedding_model = None
        self._vector_db = None
        self._model_id = "unknown"
        self._file_hashes: Dict[str, str] = {}
        self._index_dir = self.repo_root / ".lca" / "semantic_index"
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._index_dir / "index_meta.json"
        self._chunks_path = self._index_dir / "chunks.jsonl"
        self._initialize_embeddings()
        self._load_index()
    
    def _initialize_embeddings(self) -> None:
        """Initialize embedding model."""
        # Try to use sentence-transformers (local model)
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._model_id = "sentence-transformers:all-MiniLM-L6-v2"
            return
        except ImportError:
            pass
        
        # Try to use Ollama embeddings
        try:
            import httpx
            # Check if Ollama is available
            try:
                response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
                if response.status_code == 200:
                    self._embedding_model = "ollama"
                    self._model_id = "ollama:nomic-embed-text"
                    return
            except Exception:
                pass
        except ImportError:
            pass
        
        # Fallback: simple keyword-based similarity
        self._embedding_model = "keyword"
        self._model_id = "keyword"

    def _load_index(self) -> None:
        """Load persisted semantic index if compatible."""
        if not self._meta_path.exists() or not self._chunks_path.exists():
            return
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if meta.get("model_id") != self._model_id:
                return
            self._file_hashes = meta.get("file_hashes", {})
            chunks: List[CodeChunk] = []
            with self._chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    chunk = CodeChunk(
                        file_path=data["file_path"],
                        content=data["content"],
                        start_line=data["start_line"],
                        end_line=data["end_line"],
                        symbol_name=data.get("symbol_name"),
                    )
                    chunk.embedding = data.get("embedding")
                    chunks.append(chunk)
            self.chunks = chunks
        except Exception:
            self._file_hashes = {}
            self.chunks = []

    def _save_index(self) -> None:
        """Persist semantic index to disk."""
        try:
            meta = {
                "version": 1,
                "model_id": self._model_id,
                "file_hashes": self._file_hashes,
            }
            self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            with self._chunks_path.open("w", encoding="utf-8") as f:
                for chunk in self.chunks:
                    f.write(json.dumps({
                        "file_path": chunk.file_path,
                        "content": chunk.content,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "symbol_name": chunk.symbol_name,
                        "embedding": chunk.embedding,
                    }) + "\n")
        except Exception:
            pass

    def _file_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed.
        
        Returns:
            Embedding vector.
        """
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        embedding: List[float]
        
        if self._embedding_model == "ollama":
            # Use Ollama embeddings API
            try:
                import httpx
                response = httpx.post(
                    "http://127.0.0.1:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text},
                    timeout=10.0
                )
                if response.status_code == 200:
                    embedding = response.json().get("embedding", [])
                else:
                    embedding = self._keyword_embedding(text)
            except Exception:
                embedding = self._keyword_embedding(text)
        elif isinstance(self._embedding_model, object) and hasattr(self._embedding_model, 'encode'):
            # Use sentence-transformers
            embedding = self._embedding_model.encode(text).tolist()
        else:
            # Fallback to keyword-based
            embedding = self._keyword_embedding(text)
        
        # Cache embedding
        self.embeddings_cache[text_hash] = embedding
        return embedding
    
    def _keyword_embedding(self, text: str) -> List[float]:
        """
        Simple keyword-based embedding (fallback).
        
        Args:
            text: Text to embed.
        
        Returns:
            Simple embedding vector.
        """
        # Simple bag-of-words style embedding
        words = text.lower().split()
        # Create a simple frequency-based vector
        # In production, this would be replaced with proper embeddings
        unique_words = list(set(words))
        embedding = [float(words.count(w)) for w in unique_words[:100]]  # Limit to 100 dimensions
        # Pad or truncate to fixed size
        while len(embedding) < 50:
            embedding.append(0.0)
        return embedding[:50]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def index_codebase(self, chunk_size: int = 50) -> None:
        """
        Index the codebase with embeddings.
        
        Args:
            chunk_size: Number of lines per chunk.
        """
        repo_files = scan_repo(str(self.repo_root))
        python_files = [f for f in repo_files if f.endswith(".py")]

        # Remove deleted files from cache
        cached_files = set(self._file_hashes.keys())
        current_files = set(python_files)
        removed = cached_files - current_files
        if removed:
            self.chunks = [c for c in self.chunks if c.file_path not in removed]
            for f in removed:
                self._file_hashes.pop(f, None)

        for file_path in python_files:
            full_path = self.repo_root / file_path
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            content_hash = self._file_hash(content)
            if self._file_hashes.get(file_path) == content_hash:
                continue
            # Remove existing chunks for this file and re-index
            self.chunks = [c for c in self.chunks if c.file_path != file_path]
            self._index_file(file_path, chunk_size, content=content)
            self._file_hashes[file_path] = content_hash

        self._save_index()
    
    def _index_file(self, file_path: str, chunk_size: int, content: Optional[str] = None) -> None:
        """Index a single file."""
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return
        
        if content is None:
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return
        
        lines = content.splitlines()
        
        # Extract symbols for better chunking
        file_symbols = extract_symbols(full_path, use_cache=True)
        symbol_lines = {s.line - 1: s.name for s in file_symbols.symbols}  # 0-indexed
        
        # Create chunks around symbols or by line count
        chunks_created = set()
        
        # Chunk around symbols
        for line_num, symbol_name in symbol_lines.items():
            start = max(0, line_num - 5)
            end = min(len(lines), line_num + chunk_size)
            chunk_key = (start, end)
            
            if chunk_key not in chunks_created:
                chunk_content = "\n".join(lines[start:end])
                chunk = CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=start + 1,
                    end_line=end,
                    symbol_name=symbol_name,
                )
                chunk.embedding = self._get_embedding(chunk_content)
                self.chunks.append(chunk)
                chunks_created.add(chunk_key)
        
        # Fill gaps with line-based chunks
        covered_lines = set()
        for chunk in self.chunks:
            if chunk.file_path == file_path:
                covered_lines.update(range(chunk.start_line - 1, chunk.end_line))
        
        for i in range(0, len(lines), chunk_size):
            if i not in covered_lines:
                end = min(len(lines), i + chunk_size)
                chunk_content = "\n".join(lines[i:end])
                chunk = CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=i + 1,
                    end_line=end,
                )
                chunk.embedding = self._get_embedding(chunk_content)
                self.chunks.append(chunk)
    
    def search(self, query: str, top_k: int = 10) -> List[CodeMatch]:
        """
        Search codebase semantically.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
        
        Returns:
            List of code matches sorted by relevance.
        """
        if not self.chunks:
            # Index if not already indexed
            self.index_codebase()
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        matches: List[CodeMatch] = []
        for chunk in self.chunks:
            if chunk.embedding:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                match = CodeMatch(
                    file_path=chunk.file_path,
                    content=chunk.content,
                    similarity_score=similarity,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    symbol_name=chunk.symbol_name,
                    match_type="semantic",
                )
                matches.append(match)
        
        # Sort by similarity
        matches.sort(key=lambda m: -m.similarity_score)
        
        return matches[:top_k]
    
    def find_similar_code(self, code_snippet: str, top_k: int = 5) -> List[CodeMatch]:
        """
        Find code similar to a given snippet.
        
        Args:
            code_snippet: Code snippet to find similar code for.
            top_k: Number of results.
        
        Returns:
            List of similar code matches.
        """
        return self.search(code_snippet, top_k)
    
    def search_by_symbol(self, symbol_name: str) -> List[CodeMatch]:
        """
        Search for code related to a symbol.
        
        Args:
            symbol_name: Symbol name to search for.
        
        Returns:
            List of code matches.
        """
        # First, try exact symbol match
        matches: List[CodeMatch] = []
        
        for chunk in self.chunks:
            if chunk.symbol_name == symbol_name:
                match = CodeMatch(
                    file_path=chunk.file_path,
                    content=chunk.content,
                    similarity_score=1.0,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    symbol_name=symbol_name,
                    match_type="exact",
                )
                matches.append(match)
        
        # Also do semantic search
        semantic_matches = self.search(symbol_name, top_k=10)
        matches.extend(semantic_matches)
        
        # Remove duplicates and sort
        seen = set()
        unique_matches = []
        for match in matches:
            key = (match.file_path, match.start_line)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        unique_matches.sort(key=lambda m: -m.similarity_score)
        return unique_matches


def create_semantic_search(repo_root: Path) -> SemanticCodeSearch:
    """
    Create a semantic code search instance.
    
    Args:
        repo_root: Repository root directory.
    
    Returns:
        SemanticCodeSearch instance.
    """
    return SemanticCodeSearch(repo_root)


if __name__ == "__main__":
    # Demo
    import sys
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    search = create_semantic_search(repo_root)
    
    # Index codebase
    print("Indexing codebase...")
    search.index_codebase()
    print(f"Indexed {len(search.chunks)} code chunks")
    
    # Search
    query = sys.argv[2] if len(sys.argv) > 2 else "function that calculates total"
    print(f"\nSearching for: '{query}'")
    results = search.search(query, top_k=5)
    
    for i, match in enumerate(results, 1):
        print(f"\n{i}. {match.file_path}:{match.start_line} (score: {match.similarity_score:.3f})")
        print(f"   {match.content[:100]}...")
