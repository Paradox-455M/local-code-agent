from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional
import re

try:
    from memory.symbols import score_file_by_symbols, find_files_with_symbol
    from memory.dependencies import build_dependency_graph
    from memory.knowledge_graph import build_codebase_graph, CodebaseGraph
    from memory.call_graph import build_call_graph, CallGraphBuilder
    from memory.semantic_search import create_semantic_search, SemanticCodeSearch
    from memory.graph_cache import GraphCache
    from core.config import config
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    config = None  # type: ignore
    GraphCache = None  # type: ignore
    # Fallback if advanced features not available
    def score_file_by_symbols(*args, **kwargs):  # type: ignore
        return 0.0

    def find_files_with_symbol(*args, **kwargs):  # type: ignore
        return []

    def build_dependency_graph(*args, **kwargs):  # type: ignore
        return None
    
    CodebaseGraph = None  # type: ignore
    CallGraphBuilder = None  # type: ignore
    SemanticCodeSearch = None  # type: ignore
    build_codebase_graph = None  # type: ignore
    build_call_graph = None  # type: ignore
    create_semantic_search = None  # type: ignore
    ADVANCED_FEATURES_AVAILABLE = False


@dataclass
class Plan:
    goal: str
    assumptions: List[str]
    files_to_read: List[str]
    files_to_modify: List[str]
    steps: List[str]
    tests_to_run: List[str]
    risk_level: str


ALLOWED_RISK = {"low", "medium", "high"}


def _validate_repo_files(repo_files: List[str]) -> List[str]:
    return [str(Path(p)) for p in repo_files]


def _limit(items: List[str], limit: int = 10) -> List[str]:
    return items[:limit]


def _tokenize(task: str) -> List[str]:
    tokens = []
    current = []
    for ch in task.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _score_file(
    path: str, keywords: List[str], repo_root: Optional[str] = None, use_semantic: bool = True
) -> Tuple[float, List[str]]:
    """
    Score a file based on keyword matches and semantic symbol analysis.

    Args:
        path: File path to score.
        keywords: Keywords from the task.
        repo_root: Repository root for semantic analysis.
        use_semantic: Whether to use semantic (symbol-based) scoring.

    Returns:
        Tuple of (score, reasons).
    """
    p = Path(path)
    score = 0.0
    reasons = []

    if p.suffix == ".py":
        score += 1.0  # Reduced base priority for Python files
        reasons.append("python-file")

    stem = p.stem.lower()
    # Check if any keyword appears in the filename
    matched = [k for k in keywords if k and k in stem]
    
    # Boost significantly if we have a match, otherwise score stays low
    if matched:
        add = 5.0 + 1.0 * (len(matched) - 1)
        score += add
        reasons.append(f"matched:{','.join(matched)}")
    else:
        # If no keyword match, penalize or keep score low so it doesn't get picked up as "relevant" easily
        # unless it was explicitly prioritized by other means (not implemented yet)
        score -= 5.0 

    # Semantic scoring: check if file contains symbols matching keywords
    if use_semantic and repo_root and p.suffix == ".py":
        try:
            semantic_score = score_file_by_symbols(path, keywords, repo_root, use_cache=True)
            if semantic_score > 0:
                score += semantic_score * 2.0  # Weight semantic matches higher
                reasons.append(f"semantic:{semantic_score:.1f}")
        except Exception:
            # If semantic analysis fails, continue with keyword-based scoring
            pass

    # Weight by path depth to bias toward nearer files
    depth_penalty = min(len(p.parts), 5) * 0.1
    score -= depth_penalty

    return score, reasons


def _choose_files(
    task: str, repo_files: List[str], repo_root: Optional[str] = None, use_semantic: bool = True
) -> tuple[List[str], List[str], List[str]]:
    """
    Choose relevant files based on task keywords and semantic analysis.
    Automatically uses knowledge graph, call graph, and semantic search.

    Args:
        task: The task description.
        repo_files: List of all repository files.
        repo_root: Repository root for semantic analysis.
        use_semantic: Whether to use semantic (symbol-based) search.

    Returns:
        Tuple of (files_to_read, files_to_modify, rationale).
    """
    keywords = _tokenize(task)
    
    # If no keywords are found, assume it's a general task and don't select any files by default.
    if not keywords:
         return [], [], ["No relevant keywords found in task; skipping file selection."]

    # Initialize advanced features if available (automatic, no user action needed)
    knowledge_graph = None
    call_graph = None
    semantic_search = None
    
    use_kg = config is not None and getattr(config, "use_knowledge_graph", True)
    if ADVANCED_FEATURES_AVAILABLE and repo_root and use_kg:
        try:
            if GraphCache:
                cached = GraphCache(Path(repo_root)).load()
                if cached:
                    knowledge_graph, call_graph = cached
            if knowledge_graph is None:
                knowledge_graph = build_codebase_graph(Path(repo_root))
            if call_graph is None and knowledge_graph is not None:
                call_graph = build_call_graph(Path(repo_root), knowledge_graph)
            semantic_search = create_semantic_search(Path(repo_root))
            semantic_search.index_codebase()
            if GraphCache and knowledge_graph and call_graph:
                GraphCache(Path(repo_root)).save(knowledge_graph, call_graph)
        except Exception:
            pass

    # Step 1: Semantic search for relevant files (automatic)
    semantic_files: List[str] = []
    if semantic_search:
        try:
            # Use semantic search to find relevant code automatically
            semantic_matches = semantic_search.search(task, top_k=15)
            semantic_files = list(set(match.file_path for match in semantic_matches))
        except Exception:
            pass
    
    # Step 2: Symbol-based search (fallback or supplement)
    symbol_files: List[str] = []
    if use_semantic and repo_root:
        try:
            for keyword in keywords:
                if len(keyword) > 2:  # Only search for meaningful keywords
                    found = find_files_with_symbol(keyword, repo_root, use_cache=True)
                    symbol_files.extend(found)
            symbol_files = list(set(symbol_files))
        except Exception:
            pass
    
    # Step 3: Find related files using knowledge graph (automatic)
    related_files: List[str] = []
    if knowledge_graph:
        try:
            # Extract symbols from task and find related code automatically
            for keyword in keywords:
                if len(keyword) > 2:
                    # Find related code automatically
                    related_locations = knowledge_graph.find_related_code(keyword)
                    for location in related_locations:
                        if location.file_path not in related_files:
                            related_files.append(location.file_path)
            
            # If we found files via semantic search, find their related files automatically
            for file_path in semantic_files[:5]:  # Limit to top 5
                module_info = knowledge_graph.get_module_info(file_path)
                if module_info:
                    # Add imports and files that import this (automatic)
                    related_files.extend(module_info.imports[:3])  # Limit imports
                    related_files.extend(module_info.imported_by[:3])  # Limit importers
                    related_files.extend(module_info.test_files)  # Always include tests
        except Exception:
            pass
    
    # Step 4: Find callers/callees if function mentioned (automatic)
    call_related_files: List[str] = []
    if call_graph and knowledge_graph:
        try:
            # Look for function names in keywords and find callers/callees automatically
            for keyword in keywords:
                if len(keyword) > 3:  # Likely a function name
                    # Find callers automatically
                    callers = call_graph.find_callers(keyword)
                    call_related_files.extend([c.caller_file for c in callers[:5]])
                    
                    # Find callees automatically
                    callees = call_graph.find_callees(keyword)
                    # Resolve callee names to files
                    for callee_name in callees[:5]:
                        func_info = knowledge_graph.get_function_info(callee_name)
                        if func_info:
                            call_related_files.append(func_info.file_path)
        except Exception:
            pass

    # Combine all file sources (automatic discovery)
    all_candidate_files = set(semantic_files) | set(symbol_files) | set(related_files) | set(call_related_files)
    
    # Score files
    scored: List[Tuple[str, float, List[str]]] = []
    for path in all_candidate_files:
        if path not in repo_files:
            continue
        
        # We can relax the .py constraint if needed, but for now keep it to avoid selecting binary/huge files
        if not path.endswith(".py") and not path.endswith(".md"):
            continue
        
        score, reasons = _score_file(path, keywords, repo_root, use_semantic)
        
        # Boost scores based on discovery method (automatic prioritization)
        if path in semantic_files:
            score += 10.0  # High boost for semantic matches
            reasons.append("semantic-search")
        if path in symbol_files:
            score += 5.0
            reasons.append("symbol-match")
        if path in related_files:
            score += 3.0
            reasons.append("related-code")
        if path in call_related_files:
            score += 4.0
            reasons.append("call-graph")
        
        if score <= 0:
            continue
        scored.append((path, score, reasons))

    # Also score original repo files (for files not found by advanced methods)
    for path in repo_files:
        if path in all_candidate_files:
            continue  # Already scored
        
        if not path.endswith(".py") and not path.endswith(".md"):
            continue
        
        score, reasons = _score_file(path, keywords, repo_root, use_semantic)
        if score > 0:
            scored.append((path, score, reasons))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:15]  # Increased from 10 to allow more context
    
    # If we found relevant files, select them.
    if top:
        chosen = [p for p, _, _ in top]
        rationale = [f"{p} (score={s:.1f}, reasons={';'.join(r) or 'none'})" for p, s, r in top]
        return chosen, chosen, rationale
    
    # Fallback: If no files matched keywords, return empty lists instead of forcing top 5 Python files.
    # This prevents the "why are these files here?" confusion for general tasks like "say hello".
    return [], [], ["No files matched task keywords."]


def _validate_plan(plan: Plan) -> None:
    if not plan.goal or not isinstance(plan.goal, str):
        raise ValueError("goal must be non-empty string")
    if plan.risk_level not in ALLOWED_RISK:
        raise ValueError("risk_level must be one of low|medium|high")
    for field_name in ("assumptions", "files_to_read", "files_to_modify", "steps", "tests_to_run"):
        value = getattr(plan, field_name)
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list")
        if len(value) > 10:
            raise ValueError(f"{field_name} exceeds limit of 10")
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field_name} must contain strings")


def build_plan(
    task: str, repo_files: List[str], targets: List[str] | None = None, dirs: List[str] | None = None, repo_root: str | None = None
) -> Plan:
    """
    Construct a Plan object by analyzing the task and repository structure.

    Selects relevant files to read and modify based on keyword matching,
    path heuristics, and optional target/directory constraints.

    Args:
        task: The user's goal or request.
        repo_files: List of all file paths in the repository.
        targets: Optional list of specific target substrings/paths to prioritize.
        dirs: Optional list of directories to restrict the search to.

    Returns:
        A Plan object containing the goal, selected files, and steps.
    """
    repo_files = _validate_repo_files(repo_files)
    inferred = _infer_path_from_task(task)
    
    files_to_read = []
    files_to_modify = []
    rationale = []

    if inferred:
        # If the user explicitly names a file, assume they want to read/modify it?
        # Actually, for "explain main.py", we want read but NOT modify.
        # "fix main.py" -> read + modify.
        # "explain main.py" -> read only.
        
        is_modification_request = any(k in task.lower() for k in ["fix", "change", "update", "modify", "add", "remove", "delete", "create"])
        
        if is_modification_request:
            files_to_modify = [inferred]
            files_to_read = [inferred]
            rationale = [f"Inferred target from task: {inferred}"]
        else:
            # Default to read-only if no modification keywords found
            files_to_read = [inferred]
            rationale = [f"Inferred context from task: {inferred}"]
    else:
        filtered = _filter_by_dirs(repo_files, dirs) if dirs else repo_files
        prioritized = _prioritize_targets(filtered, targets)
        # Use semantic search if available
        files_to_read, files_to_modify, rationale = _choose_files(task, prioritized, repo_root=repo_root, use_semantic=True)
    # Enforce caps before validation to avoid ValueError when selection exceeds limits
    files_to_read = _limit(files_to_read, limit=10)
    files_to_modify = _limit(files_to_modify, limit=10)

    if files_to_modify:
        steps = [
            "Inspect relevant files",
            "Draft diffs for requested changes",
            "Review diffs and collect risks",
        ]
    elif files_to_read:
        steps = [
            "Read context files",
            "Analyze code structure",
            "Answer user query",
        ]
    else:
        steps = [
            "Process user request",
            "Generate response",
        ]

    plan = Plan(
        goal=task,
        assumptions=[
            "Offline Ollama-only environment",
            "Human will approve before apply",
            "File selection rationale: " + "; ".join(rationale) if rationale else "File selection fallback to default Python files.",
        ],
        files_to_read=files_to_read,
        files_to_modify=files_to_modify,
        steps=steps,
        tests_to_run=["python -m py_compile *.py"],
        risk_level="medium",
    )
    _validate_plan(plan)
    return plan


def to_json(plan: Plan) -> str:
    """Serialize the Plan object to a JSON string."""
    return json.dumps(asdict(plan), indent=2)


def _prioritize_targets(repo_files: List[str], targets: List[str] | None) -> List[str]:
    if not targets:
        return repo_files
    target_set = set(targets)
    prioritized = [p for p in repo_files if any(t in p for t in target_set)]
    rest = [p for p in repo_files if p not in prioritized]
    return prioritized + rest


def _infer_path_from_task(task: str) -> Optional[str]:
    lower = task.lower()
    home = Path.home()
    name = _extract_filename(task)

    # Only return if we found a concrete .py file token
    if not name:
        return None

    # Handle absolute and tilde-expansion
    if name.startswith("/"):
        return str(Path(name).expanduser().resolve())
    if name.startswith("~"):
        return str(Path(name).expanduser().resolve())

    if "desktop" in lower:
        desktop = home / "Desktop"
        return str((desktop / name).resolve())

    if "home" in lower:
        return str((home / name).resolve())

    # Only return CWD relative path if the file actually exists or user explicitly wants to create it
    # For now, let's assume if they named a file, they mean it.
    return str((Path.cwd() / name).resolve())


def _extract_filename(text: str) -> Optional[str]:
    tokens = text.replace(",", " ").replace(";", " ").split()
    for token in tokens:
        tok = token.strip().strip(",.")
        if tok.endswith(".py"):
            return tok
    return None


def _filter_by_dirs(repo_files: List[str], dirs: List[str]) -> List[str]:
    prefixes = [d.rstrip("/") for d in dirs if d.strip()]
    return [p for p in repo_files if any(Path(p).as_posix().startswith(prefix) for prefix in prefixes)]


if __name__ == "__main__":
    sample_files = ["core/config.py", "core/llm.py", "main.py"]
    demo_plan = build_plan("Adjust model config", sample_files)
    print(to_json(demo_plan))
