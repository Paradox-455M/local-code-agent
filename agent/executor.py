from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from tools.diff import unified_diff
from core.llm import ask

try:
    from memory.symbols import extract_symbols, Symbol
    from memory.dependencies import build_dependency_graph
    from agent.prompt_engineer import get_task_type_prompt, get_prompt_with_examples, get_chain_of_thought_prompt
    from agent.task_classifier import classify_task, TaskType
    from agent.context_builder import build_intelligent_context, IntelligentContextBuilder
    from agent.refiner import create_refiner, CodeRefiner
    from memory.project_conventions import learn_project_conventions, ProjectConventionLearner
    from agent.performance import ContextPruner, get_global_monitor, LazyFileLoader
    from agent.templates import find_template_for_task
    TEMPLATES_AVAILABLE = True
    PERFORMANCE_AVAILABLE = True
except ImportError:
    # Fallback if symbols module not available
    def extract_symbols(*args, **kwargs):  # type: ignore
        return None

    Symbol = None  # type: ignore
    build_dependency_graph = None  # type: ignore
    get_task_type_prompt = None  # type: ignore
    get_prompt_with_examples = None  # type: ignore
    get_chain_of_thought_prompt = None  # type: ignore
    classify_task = None  # type: ignore
    TaskType = None  # type: ignore
    build_intelligent_context = None  # type: ignore
    IntelligentContextBuilder = None  # type: ignore
    create_refiner = None  # type: ignore
    CodeRefiner = None  # type: ignore
    learn_project_conventions = None  # type: ignore
    ProjectConventionLearner = None  # type: ignore
    ContextPruner = None  # type: ignore
    get_global_monitor = None  # type: ignore
    LazyFileLoader = None  # type: ignore
    find_template_for_task = None  # type: ignore
    TEMPLATES_AVAILABLE = False
    PERFORMANCE_AVAILABLE = False


@dataclass
class ExecutorResult:
    diffs: List[Dict[str, str]]
    notes: List[str]
    confidence: float
    prompt_excerpt: str | None = None
    llm_excerpt: str | None = None
    context_preview: List[Dict[str, str]] | None = None


def _read_files(base: Path, files: List[str], parallel: bool = True) -> Dict[str, str]:
    """
    Read multiple files, optionally in parallel for better performance.
    
    Args:
        base: Base directory path.
        files: List of relative file paths.
        parallel: Whether to read files in parallel (default: True).
    
    Returns:
        Dictionary mapping file paths to their contents.
    """
    contents: Dict[str, str] = {}
    use_parallel = parallel and len(files) > 3

    if use_parallel:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def read_file(rel_path: str) -> tuple[str, Optional[str]]:
                path = base / rel_path
                if not path.exists():
                    return rel_path, None
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    return rel_path, content
                except Exception:
                    return rel_path, None

            with ThreadPoolExecutor(max_workers=min(8, len(files))) as executor:
                futures = {executor.submit(read_file, rel_path): rel_path for rel_path in files}
                for future in as_completed(futures):
                    rel_path, content = future.result()
                    if content is not None:
                        contents[rel_path] = content
            return contents
        except ImportError:
            pass

    # Sequential reading (default for few files, or fallback)
    for rel_path in files:
        path = base / rel_path
        if not path.exists():
            continue
        try:
            contents[rel_path] = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
    return contents


def execute(
    task: str,
    files_to_read: List[str],
    files_to_modify: List[str],
    repo_root: str,
    instruction: Optional[str] = None,
    context_paths: Optional[List[str]] = None,
    max_context_files: int = 5,
    language: Optional[str] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    model: Optional[str] = None,
    mode: Optional[str] = None,
    context_cache: Optional[Dict[str, str]] = None,
    symbols: Optional[List[str]] = None,
    relaxed_validation: bool = False,
    enable_llm: bool = False,
    use_intelligent_context: Optional[bool] = None,
) -> ExecutorResult:
    """
    Produce diffs for requested files based on task and context.

    Strategy:
    1. Try to match a deterministic instruction (replace/append/insert).
    2. If no match, use the LLM to synthesize unified diffs from task + context.
    3. Validate and clean up generated diffs.

    Args:
        task: The natural language goal.
        files_to_read: List of file paths to read as context.
        files_to_modify: List of file paths allowed to be modified.
        repo_root: Absolute path to the repository root.
        instruction: Optional explicit instruction (overrides task inference).
        context_paths: Additional context files.
        max_context_files: Maximum number of files to include in LLM prompt.
        language: Optional language hint.
        llm_fn: Optional callable for LLM (for testing/mocking).
        model: Model name to use.
        mode: Execution mode (bugfix, refactor, etc.).
        context_cache: Cache for file contents.
        symbols: List of symbols to prioritize in context.
        relaxed_validation: If True, loosen safety checks on diffs.
        enable_llm: Whether to allow LLM usage if deterministic match fails.

    Returns:
        ExecutorResult containing diffs, confidence score, and notes.
    """
    base = Path(repo_root)
    existing_files = _read_files(base, files_to_read, parallel=True)

    reasons: List[str] = []
    clarifications: List[str] = []

    if not files_to_modify:
        reasons.append("No target files specified for modification.")
        clarifications.append("Which files should change and what should be updated?")

    if not existing_files:
        reasons.append("Requested context files could not be read or were empty selection.")
        clarifications.append("Provide concrete files to inspect and desired edits.")

    parsed = _parse_instruction(instruction or task)
    if not parsed:
        clarifications.append(
            "Specify an instruction like: replace 'old' with 'new', "
            "append 'text' after 'anchor', or insert 'text' before 'anchor'. "
            "Otherwise the LLM will attempt code generation from context."
        )

    if reasons:
        notes = reasons + [f"Clarification needed: {q}" for q in clarifications]
        return ExecutorResult(diffs=[], notes=notes, confidence=0.0)

    # 1) Deterministic path if instruction matches known pattern.
    if parsed:
        deterministic = _deterministic_apply(parsed, existing_files, files_to_modify)
        if deterministic.diffs:
            return deterministic

    # 2) LLM path for code synthesis (explicitly enabled).
    if llm_fn is None and not enable_llm:
        notes = [
            "No deterministic instruction matched and LLM is disabled.",
            "Clarification needed: provide an explicit instruction (replace/append/insert), "
            "or run via the CLI which enables the local LLM.",
        ]
        return ExecutorResult(diffs=[], notes=notes, confidence=0.0)

    llm_callable = llm_fn or (lambda p: ask(p, model=model))

    # Use intelligent context when available and not explicitly disabled (e.g. -f explicit files)
    use_intelligent_context = use_intelligent_context if use_intelligent_context is not None else True
    use_intelligent_context = use_intelligent_context and build_intelligent_context is not None
    if use_intelligent_context:
        try:
            context_obj = build_intelligent_context(
                repo_root=base,
                task=task,
                files=context_paths or files_to_read,
                symbols=symbols,
                max_files=max_context_files,
                max_bytes=40000,
                include_tests=True,
                include_related=True,
            )
            snippets = context_obj.to_dict_list()
        except Exception:
            # Fall back to basic context building
            use_intelligent_context = False
    
    if not use_intelligent_context:
        snippets = build_context(
            base,
            context_paths or files_to_read,
            max_context_files=max_context_files,
            cache=context_cache,
            symbols=symbols,
        )
    # Load project conventions for prompt enhancement
    conventions_text = None
    if ProjectConventionLearner:
        try:
            learner = ProjectConventionLearner(base)
            rules_file = base / ".lca-rules.yaml"
            if rules_file.exists():
                learner.load_rules(rules_file)
            else:
                learner.learn_from_codebase(sample_size=20)
            
            # Build conventions text for prompt
            conv = learner.conventions
            conventions_text = f"""Code Style:
- Quote style: {conv.style.quote_style}
- Max line length: {conv.style.max_line_length}
- Indent: {conv.style.indent_size} {conv.style.indent_style}
- Function naming: {conv.naming.function_naming}
- Class naming: {conv.naming.class_naming}"""
        except Exception:
            pass
    
    # Performance monitoring for prompt building
    if PERFORMANCE_AVAILABLE and get_global_monitor:
        monitor = get_global_monitor()
        monitor.start("prompt_building")
    
    # Determine if task is complex (use CoT for multi-file refactors/creates or long tasks)
    task_type_for_cot = None
    if classify_task is not None:
        try:
            task_type_for_cot, _ = classify_task(task)
        except Exception:
            task_type_for_cot = None
    is_multi_file = len(files_to_modify) > 2
    is_long_task = len(task.split()) > 25
    refactor_or_create = (TaskType.REFACTOR, TaskType.CREATE) if TaskType is not None else ()
    is_complex_type = task_type_for_cot in refactor_or_create
    is_complex = (is_multi_file and is_complex_type) or len(files_to_modify) > 3 or is_long_task
    use_cot = is_complex and get_chain_of_thought_prompt is not None

    prompt = _build_prompt(
        task,
        instruction,
        snippets,
        language,
        mode,
        use_enhanced_prompts=True,
        use_few_shot=True,
        use_chain_of_thought=use_cot,
        conventions=conventions_text,
        files_to_modify=files_to_modify,
    )
    # End prompt building monitoring
    if PERFORMANCE_AVAILABLE and get_global_monitor:
        monitor = get_global_monitor()
        monitor.end("prompt_building")
        monitor.start("llm_call")
    
    try:
        llm_output = llm_callable(prompt)
    except Exception as exc:  # pragma: no cover - network or model issues
        if PERFORMANCE_AVAILABLE and get_global_monitor:
            monitor = get_global_monitor()
            monitor.end("llm_call")
        notes = [f"LLM call failed: {exc}", "Clarification needed or retry with explicit instruction."]
        return ExecutorResult(diffs=[], notes=notes, confidence=0.0)
    
    # End LLM call monitoring
    if PERFORMANCE_AVAILABLE and get_global_monitor:
        monitor = get_global_monitor()
        monitor.end("llm_call")

    diffs = _extract_diffs(llm_output)
    if not diffs:
        strict_prompt = (
            prompt + "\n\n[IMPORTANT] Return ONLY the unified diff(s). "
            "No markdown, no explanations, no prose. Start with --- path and +++ path, use @@ hunks."
        )
        try:
            strict_output = llm_callable(strict_prompt)
            diffs = _extract_diffs(strict_output)
            llm_output = strict_output
        except Exception:
            diffs = []

    if not diffs:
        notes = [
            "LLM did not return a valid unified diff after retry.",
            "Clarification needed: provide exact change or anchors, or narrow file list.",
        ]
        return ExecutorResult(diffs=[], notes=notes, confidence=0.0)

    diffs = _attach_paths(diffs)
    valid_diffs = _validate_diffs(
        diffs,
        relaxed=relaxed_validation,
        allowed_paths=files_to_modify,
        repo_root=repo_root,
    )

    # Validation rejected diffs: retry once with explicit feedback
    if not valid_diffs and diffs and not relaxed_validation:
        feedback_prompt = (
            prompt + "\n\n[RETRY] Your previous output had invalid diffs "
            "(e.g. wrong format, disallowed paths, or missing @@ hunks). "
            "Allowed files to modify: " + (", ".join(files_to_modify) if files_to_modify else "see context") + ". "
            "Output ONLY valid unified diffs. Use ---/+++ and @@ -start,count +start,count @@."
        )
        try:
            feedback_output = llm_callable(feedback_prompt)
            diffs = _attach_paths(_extract_diffs(feedback_output))
            valid_diffs = _validate_diffs(
                diffs,
                relaxed=False,
                allowed_paths=files_to_modify,
                repo_root=repo_root,
            )
            if valid_diffs:
                llm_output = feedback_output
        except Exception:
            pass

    if not valid_diffs:
        notes = [
            "Diffs were malformed or exceeded safety limits.",
            "Clarification needed: provide smaller, well-formed changes.",
        ]
        return ExecutorResult(
            diffs=[],
            notes=notes,
            confidence=0.0,
            prompt_excerpt=_excerpt(prompt),
            llm_excerpt=_excerpt(llm_output),
            context_preview=snippets,
        )

    notes_list: List[str] = ["Generated diffs via LLM"]
    if ProjectConventionLearner:
        try:
            learner = ProjectConventionLearner(Path(repo_root))
            rules_file = Path(repo_root) / ".lca-rules.yaml"
            if rules_file.exists():
                learner.load_rules(rules_file)
            else:
                learner.learn_from_codebase(sample_size=20)
            for diff in valid_diffs:
                file_path = diff.get("path", "")
                if file_path:
                    full_path = Path(repo_root) / file_path
                    if full_path.exists():
                        content = full_path.read_text(encoding="utf-8", errors="replace")
                        _, violations = learner.enforce_conventions(content)
                        if violations:
                            notes_list.append(
                                f"Convention check for {file_path}: {len(violations)} potential issues"
                            )
        except Exception:
            pass

    return ExecutorResult(
        diffs=valid_diffs,
        notes=notes_list,
        confidence=0.6,
        prompt_excerpt=_excerpt(prompt),
        llm_excerpt=_excerpt(llm_output),
        context_preview=snippets,
    )


def to_json(result: ExecutorResult) -> str:
    """Serialize ExecutorResult to a JSON string."""
    data: Dict[str, Any] = asdict(result)
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    sample_result = execute(
        task="replace foo with bar",
        files_to_read=["core/config.py"],
        files_to_modify=["core/config.py"],
        repo_root=".",
    )
    print(to_json(sample_result))


def _parse_instruction(text: str) -> Optional[Dict[str, str]]:
    text = text.strip()
    if not text:
        return None

    replace_match = re.search(
        r"replace\s+['\"]?(?P<old>.+?)['\"]?\s+with\s+['\"]?(?P<new>.+)", text, flags=re.IGNORECASE
    )
    if replace_match:
        old = replace_match.group("old").strip()
        new = replace_match.group("new").strip()
        if old and new:
            return {"kind": "replace", "old": old, "new": new}

    append_match = re.search(
        r"append\s+['\"]?(?P<text>.+?)['\"]?\s+after\s+['\"]?(?P<anchor>.+)", text, flags=re.IGNORECASE
    )
    if append_match:
        insert_text = append_match.group("text").strip()
        anchor = append_match.group("anchor").strip()
        if insert_text and anchor:
            return {"kind": "append_after", "text": insert_text, "anchor": anchor}

    insert_match = re.search(
        r"(insert|prepend)\s+['\"]?(?P<text>.+?)['\"]?\s+(?:before|above)\s+['\"]?(?P<anchor>.+)",
        text,
        flags=re.IGNORECASE,
    )
    if insert_match:
        insert_text = insert_match.group("text").strip()
        anchor = insert_match.group("anchor").strip()
        if insert_text and anchor:
            return {"kind": "insert_before", "text": insert_text, "anchor": anchor}

    return None


def _deterministic_apply(parsed: Dict[str, str], existing_files: Dict[str, str], files_to_modify: List[str]) -> ExecutorResult:
    kind = parsed["kind"]
    old_text = parsed.get("old", "")
    new_text = parsed.get("new", "")
    anchor = parsed.get("anchor", "")
    insert_text = parsed.get("text", "")

    diffs: List[Dict[str, str]] = []
    applied = 0
    for rel_path in files_to_modify:
        if rel_path not in existing_files:
            continue
        original = existing_files[rel_path]
        updated = None

        if kind == "replace":
            if old_text in original:
                candidate = original.replace(old_text, new_text, 1)
                if candidate != original:
                    updated = candidate
        elif kind == "append_after":
            updated = _append_after(original, anchor, insert_text)
        elif kind == "insert_before":
            updated = _insert_before(original, anchor, insert_text)

        if updated and updated != original:
            applied += 1
            diff_text = unified_diff(original, updated, rel_path)
            diffs.append({"path": rel_path, "diff": diff_text})

    if not diffs:
        notes = [
            "Instruction did not match anchors or content in target files.",
            "Clarification needed: confirm the exact text/anchors and file path.",
        ]
        return ExecutorResult(diffs=[], notes=notes, confidence=0.0)

    notes = [f"Applied {kind} in {applied} file(s)."]
    confidence = 0.7 if applied else 0.0
    return ExecutorResult(diffs=diffs, notes=notes, confidence=confidence)


def _append_after(original: str, anchor: str, insert_text: str) -> Optional[str]:
    lines = original.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if anchor in line:
            lines.insert(idx + 1, insert_text + ("\n" if not insert_text.endswith("\n") else ""))
            return "".join(lines)
    return None


def _insert_before(original: str, anchor: str, insert_text: str) -> Optional[str]:
    lines = original.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if anchor in line:
            lines.insert(idx, insert_text + ("\n" if not insert_text.endswith("\n") else ""))
            return "".join(lines)
    return None


def build_context(
    base: Path,
    files: List[str],
    max_context_files: int = 5,
    max_bytes: int = 4000,
    cache: Optional[Dict[str, str]] = None,
    symbols: Optional[List[str]] = None,
    include_related: bool = True,
) -> List[Dict[str, str]]:
    """
    Build context snippets from files, with intelligent selection and related file inclusion.

    Args:
        base: Base path for resolving file paths.
        files: List of file paths (relative to base).
        max_context_files: Maximum number of files to include.
        max_bytes: Maximum bytes per snippet.
        cache: Optional cache for file contents.
        symbols: Optional list of symbols to prioritize.
        include_related: Whether to include related files (imports, dependencies).

    Returns:
        List of context snippets with path and content.
    """
    cache = cache if cache is not None else {}
    sym_lower = [s.lower() for s in symbols] if symbols else []

    # Build dependency graph if including related files
    related_files: Set[str] = set()
    if include_related and build_dependency_graph:
        try:
            dep_graph = build_dependency_graph(str(base))
            for file_path in files:
                if file_path.endswith(".py"):
                    related = dep_graph.get_related_files(file_path, max_depth=1)
                    related_files.update(related)
        except Exception:
            # If dependency analysis fails, continue without related files
            pass

    # Combine original files with related files
    all_files = list(set(files) | related_files)
    
    # Score and rank files
    ranked: List[tuple[str, float, int, int]] = []  # (path, score, size, path_length)
    for rel_path in all_files:
        path = base / rel_path
        if not path.exists() or not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue

        score = 0.0
        lp = rel_path.lower()

        # Base score: original files are more important than related files
        if rel_path in files:
            score += 10.0
        else:
            score += 2.0  # Related files get lower base score

        # Symbol-based scoring
        if sym_lower:
            # Path-based scoring
            path_score = 2 * sum(1 for s in sym_lower if s in lp)
            score += path_score

            # Semantic symbol scoring
            if rel_path.endswith(".py"):
                try:
                    from memory.symbols import score_file_by_symbols
                    semantic_score = score_file_by_symbols(rel_path, symbols or [], str(base), use_cache=True)
                    score += semantic_score * 3.0  # Weight semantic matches highly
                except Exception:
                    pass

        ranked.append((rel_path, score, size, len(rel_path)))

    # Sort by score (descending), then by size (ascending), then by path length
    ranked.sort(key=lambda t: (-t[1], t[2], t[3]))

    # Select top files
    selected_files = [rel_path for rel_path, _, _, _ in ranked[:max_context_files]]

    # Build snippets with smart truncation
    snippets: List[Dict[str, str]] = []
    total_bytes = 0
    max_total_bytes = max_bytes * max_context_files  # Total budget

    for rel_path in selected_files:
        path = base / rel_path
        if rel_path in cache:
            data = cache[rel_path]
        else:
            try:
                data = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            cache[rel_path] = data

        # Calculate available bytes (distribute remaining budget)
        remaining_files = max_context_files - len(snippets)
        available_bytes = min(max_bytes, (max_total_bytes - total_bytes) // max(1, remaining_files))

        snippet = _select_snippet(
            data, symbols=sym_lower, max_bytes=available_bytes, file_path=str(path), repo_root=str(base)
        )
        snippet = _extract_structural_snippet(snippet, data)
        
        # Smart truncation: if snippet is still too long, truncate intelligently
        snippet_bytes = len(snippet.encode("utf-8"))
        if snippet_bytes > available_bytes:
            # Truncate at line boundaries
            lines = snippet.splitlines()
            truncated_lines = []
            current_bytes = 0
            for line in lines:
                line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
                if current_bytes + line_bytes > available_bytes:
                    break
                truncated_lines.append(line)
                current_bytes += line_bytes
            snippet = "\n".join(truncated_lines)
            if truncated_lines:
                snippet += "\n... (truncated)"

        snippets.append({"path": rel_path, "snippet": snippet})
        total_bytes += len(snippet.encode("utf-8"))

    return snippets


def _build_prompt(
    task: str,
    instruction: Optional[str],
    snippets: List[Dict[str, Any]],
    language: Optional[str],
    mode: Optional[str],
    use_enhanced_prompts: bool = True,
    use_few_shot: bool = True,
    use_chain_of_thought: bool = False,
    conventions: Optional[str] = None,
    files_to_modify: Optional[List[str]] = None,
) -> str:
    """
    Build a prompt for the LLM to generate diffs with enhanced prompt engineering.
    """
    mode = mode or _infer_mode(task)

    # Build context string from snippets; include [test] / [related] etc. when available
    def _block(s: Dict[str, Any]) -> str:
        path = s.get("path", "")
        typ = s.get("type", "")
        suffix = f" [{typ}]" if typ and typ != "code" else ""
        return f"--- {path}{suffix} ---\n{s.get('snippet', '')}"

    snippet_blocks = "\n".join(_block(s) for s in snippets)

    # Allowed-files line (reduces wrong-file edits)
    allowed_files_line = ""
    if files_to_modify:
        allowed_files_line = (
            f"\n\nAllowed files to modify (only these): {', '.join(files_to_modify)}. "
            "Use these exact paths in ---/+++ headers."
        )

    def _append_format_and_return(base: str) -> str:
        if instruction:
            base += f"\n\nAdditional instruction: {instruction}"
        base += allowed_files_line
        if mode:
            base += f"\n\nMode: {mode}"
        lang_line = f"Language/framework hint: {language}" if language else "Language/framework hint: unknown"
        base += f"\n{lang_line}"
        base += "\n\nGenerate a unified diff for each file that needs changes."
        base += "\nFormat: --- path/to/file\n+++ path/to/file"
        base += "\n@@ -start,count +start,count @@\n context\n-old\n+new"
        base += "\n\nDo not modify test files unless explicitly requested."
        base += "\n\nOutput only the unified diff(s). No markdown, no prose before or after."
        if "create" in task.lower() and instruction is None:
            base += "\nIf unsure, propose a minimal stub with clear anchors."
        return base

    # Template-matched task (add function, fix bug, refactor, add test, update docs)
    if use_enhanced_prompts and TEMPLATES_AVAILABLE and find_template_for_task is not None:
        tpl = find_template_for_task(task)
        if tpl is not None:
            try:
                base = tpl.generate_prompt(task, context=snippet_blocks)
                return _append_format_and_return(base)
            except Exception:
                pass

    # Use enhanced prompts if available
    if use_enhanced_prompts and classify_task is not None and get_task_type_prompt is not None:
        try:
            task_type, _ = classify_task(task)

            if use_chain_of_thought and get_chain_of_thought_prompt is not None:
                base_prompt = get_chain_of_thought_prompt(task, snippet_blocks, is_complex=True)
            elif use_few_shot and get_prompt_with_examples is not None:
                base_prompt = get_prompt_with_examples(
                    task, task_type, snippet_blocks, conventions=conventions
                )
            else:
                base_prompt = get_task_type_prompt(task, task_type, snippet_blocks, conventions)

            return _append_format_and_return(base_prompt)
        except Exception:
            pass

    # Basic prompt (fallback)
    anchor_hint = ""
    if "create" in task.lower() and instruction is None:
        anchor_hint = "\nIf unsure, propose a minimal stub with clear anchors."
    lang_line = f"Language/framework hint: {language}" if language else "Language/framework hint: unknown"
    return (
        "You are an offline coding assistant. Work in two steps: (1) outline a brief plan, "
        "(2) produce unified diffs only. If unsure, return no diff.\n"
        f"Task: {task}\n"
        f"Instruction: {instruction or 'none'}\n"
        f"Mode: {mode}\n"
        f"{lang_line}\n"
        + (f"Allowed files to modify: {', '.join(files_to_modify)}.\n" if files_to_modify else "")
        + "Guardrails: Do not modify tests unless explicitly requested. Keep changes minimal and focused.\n"
        "Context snippets:\n"
        f"{snippet_blocks}\n"
        "Plan format: 2–3 rationale bullets, then unified diff(s) with ---/+++ headers. No extra text after diffs."
        f"{anchor_hint}"
    )


def _extract_diffs_from_text(raw: str) -> List[Dict[str, str]]:
    """Parse unified diffs from raw text (line-by-line)."""
    diffs: List[Dict[str, str]] = []
    lines = raw.splitlines()
    current: List[str] = []
    started = False

    for line in lines:
        if line.startswith("--- "):
            if current:
                diffs.append({"path": "", "diff": "\n".join(current)})
                current = []
            started = True
        if current or line.startswith("--- "):
            current.append(line)
        elif started:
            current.append(line)

    if current:
        diffs.append({"path": "", "diff": "\n".join(current)})

    valid = []
    for d in diffs:
        if "--- " in d["diff"] and "+++" in d["diff"] and "@@" in d["diff"]:
            valid.append(d)
    return valid


def _extract_diffs(text: str) -> List[Dict[str, str]]:
    """Extract unified diffs from LLM output. Tries raw text first, then ```diff or ``` blocks."""
    out = _extract_diffs_from_text(text)
    if out:
        return out

    # Fallback: extract from ```diff ... ``` or ``` ... ``` code blocks
    for pattern in (r"```(?:diff)?\s*\n?(.*?)```", r"```\s*\n?(.*?)```"):
        for m in re.finditer(pattern, text, re.DOTALL):
            block = m.group(1).strip()
            if "--- " in block and "+++" in block and "@@" in block:
                out = _extract_diffs_from_text(block)
                if out:
                    return out
    return []


def _validate_diffs(
    diffs: List[Dict[str, str]],
    max_lines: int = 8000,
    max_bytes: int = 2_000_000,
    relaxed: bool = False,
    allowed_paths: Optional[List[str]] = None,
    repo_root: Optional[str] = None,
) -> List[Dict[str, str]]:
    validated: List[Dict[str, str]] = []
    allowed_set = {Path(p).as_posix() for p in (allowed_paths or [])}
    repo_root_path = Path(repo_root).resolve() if repo_root else None
    for d in diffs:
        diff_text = d.get("diff", "")
        if not diff_text.strip():
            continue
        if len(diff_text) > max_bytes:
            continue
        line_count = diff_text.count("\n") + 1
        if line_count > max_lines and not relaxed:
            continue
        # Basic sanity: starts with --- and contains hunk marker
        if not diff_text.strip().startswith("--- "):
            continue
        if "@@" not in diff_text:
            continue
        path = (d.get("path") or "").strip()
        if not path:
            # Infer from diff headers if missing.
            old_path = ""
            new_path = ""
            for line in diff_text.splitlines():
                if line.startswith("--- "):
                    old_path = line[4:].strip().split("\t")[0]
                elif line.startswith("+++ "):
                    new_path = line[4:].strip().split("\t")[0]
                    break
            candidate = new_path if new_path and new_path != "/dev/null" else old_path
            if candidate:
                if candidate.startswith("a/") or candidate.startswith("b/"):
                    candidate = candidate[2:]
                path = candidate
        if not path and not relaxed:
            continue
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        if allowed_set and path and path not in allowed_set and Path(path).as_posix() not in allowed_set and not relaxed:
            continue
        if repo_root_path:
            path_obj = Path(path)
            resolved = path_obj if path_obj.is_absolute() else (repo_root_path / path_obj)
            resolved = resolved.resolve()
            try:
                if not resolved.is_relative_to(repo_root_path) and not relaxed:
                    continue
            except AttributeError:
                # Python <3.9 compatibility fallback
                if not str(resolved).startswith(str(repo_root_path)) and not relaxed:
                    continue
        validated.append({"path": path, "diff": diff_text})
    return validated


def _infer_mode(task: str) -> str:
    lowered = task.lower()
    if any(k in lowered for k in ["bug", "fix", "error", "issue"]):
        return "bugfix"
    if any(k in lowered for k in ["refactor", "cleanup", "rename"]):
        return "refactor"
    if any(k in lowered for k in ["add", "implement", "feature", "new"]):
        return "feature"
    return "general"


def _extract_structural_snippet(snippet: str, full_text: str) -> str:
    """
    Lightweight heuristic: if we can find def/class lines in the first 2000 chars,
    include them to give LLM more structure; otherwise return the original snippet.
    """
    head = full_text[:2000]
    lines = head.splitlines()
    struct_lines = [ln for ln in lines if ln.lstrip().startswith(("def ", "class "))]
    if struct_lines:
        return snippet + "\n\n# structure:\n" + "\n".join(struct_lines[:20])
    return snippet


def _select_snippet(
    data: str, symbols: Optional[List[str]], max_bytes: int, file_path: Optional[str] = None, repo_root: Optional[str] = None
) -> str:
    """
    Select the most relevant snippet from file content.

    Prioritizes snippets containing task-relevant symbols.

    Args:
        data: Full file content.
        symbols: List of symbol names to prioritize.
        max_bytes: Maximum bytes for the snippet.
        file_path: Path to the file (for semantic analysis).
        repo_root: Repository root (for semantic analysis).

    Returns:
        Selected snippet string.
    """
    if not symbols:
        return data[:max_bytes]

    # Try semantic-based selection first
    if file_path and repo_root:
        try:
            file_symbols = extract_symbols(file_path, use_cache=True)
            if file_symbols and file_symbols.symbols:
                # Find symbols matching our search terms
                symbol_names_lower = [s.lower() for s in symbols]
                matching_symbols = [
                    s for s in file_symbols.symbols if s.name.lower() in symbol_names_lower or any(
                        sym.lower() in s.name.lower() for sym in symbols
                    )
                ]

                if matching_symbols:
                    # Select snippet around the first matching symbol
                    first_match = matching_symbols[0]
                    # Get context around the symbol (include function/class definition)
                    start_line = max(0, first_match.line - 10)
                    lines = data.splitlines()
                    if start_line < len(lines):
                        # Calculate byte position
                        start_bytes = len("\n".join(lines[:start_line])) + (1 if start_line > 0 else 0)
                        end_bytes = min(start_bytes + max_bytes, len(data))
                        snippet = data[start_bytes:end_bytes]

                        # If snippet is too short, try to include more context
                        if len(snippet) < max_bytes // 2 and len(matching_symbols) > 1:
                            # Include multiple symbols
                            last_match = matching_symbols[-1]
                            end_line = min(len(lines), last_match.line + 20)
                            end_bytes = len("\n".join(lines[:end_line])) + (1 if end_line > 0 else 0)
                            snippet = data[start_bytes:min(end_bytes, start_bytes + max_bytes)]

                        return snippet
        except Exception:
            # Fallback to simple text search if semantic analysis fails
            pass

    # Fallback: Simple text search
    lower = data.lower()
    for sym in symbols:
        idx = lower.find(sym.lower())
        if idx != -1:
            start = max(idx - max_bytes // 3, 0)
            end = min(idx + max_bytes // 2, len(data))
            return data[start:end]

    return data[:max_bytes]


def _excerpt(text: str, limit: int = 600) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _attach_paths(diffs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def _normalize_path_token(raw: str) -> str:
        token = raw.split("\t")[0].strip()
        if token.startswith("a/") or token.startswith("b/"):
            token = token[2:]
        return token

    attached = []
    for d in diffs:
        diff_text = d.get("diff", "")
        path = d.get("path") or ""
        if not path:
            old_path = ""
            new_path = ""
            for line in diff_text.splitlines():
                if line.startswith("--- "):
                    old_path = _normalize_path_token(line[4:])
                elif line.startswith("+++ "):
                    new_path = _normalize_path_token(line[4:])
                    # Prefer the new path when available.
                    break
            if new_path and new_path != "/dev/null":
                path = new_path
            elif old_path and old_path != "/dev/null":
                path = old_path
        attached.append({"path": path, "diff": diff_text})
    return attached
