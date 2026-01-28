from __future__ import annotations

import json
import re
import select
import subprocess
import sys
import time
import tty
import termios
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from rich.live import Live

from agent.executor import execute, to_json as executor_to_json
from agent.planner import build_plan, to_json as plan_to_json
from agent.reviewer import review_diffs, to_json as reviewer_to_json
from agent.task_classifier import classify_task, is_conversational_query, should_modify_files
from agent.decomposer import decompose_task, DecomposedTask, SubtaskStatus, get_ready_subtasks
from agent.validator import validate_before_apply, ValidationResult
from core.config import config
from core.config_file import create_default_config, load_config, find_config_file, validate_config
from core.llm import ask, ask_stream
from memory.index import scan_repo
from memory.patterns import PatternMatcher, learn_from_task
from memory.conversation import get_conversation_manager
from memory.feedback import collect_feedback, analyze_feedback_patterns
from tools.apply_patch import apply_patch, PatchError
from core.exceptions import LocalCodeAgentError
from typer.models import OptionInfo

# Sprint 3 imports
try:
    from memory.knowledge_graph import build_codebase_graph
    from memory.call_graph import build_call_graph
    from memory.semantic_search import create_semantic_search
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False

# Sprint 4 imports
try:
    from agent.refactor import create_refactoring_tool, RefactoringTool
    from agent.refiner import create_refiner
    from memory.project_conventions import learn_project_conventions, ProjectConventionLearner
    REFACTORING_AVAILABLE = True
except ImportError:
    REFACTORING_AVAILABLE = False
    RefactoringTool = None  # type: ignore
    create_refactoring_tool = None  # type: ignore
    create_refiner = None  # type: ignore
    ProjectConventionLearner = None  # type: ignore
    learn_project_conventions = None  # type: ignore

# Sprint 5 imports
try:
    from agent.test_intelligence import create_test_intelligence, TestIntelligence
    from agent.test_runner import create_test_runner, TestRunner
    from agent.error_analyzer import create_error_analyzer, ErrorAnalyzer
    from agent.debugger import create_debugger, Debugger
    TEST_INTELLIGENCE_AVAILABLE = True
except ImportError:
    TEST_INTELLIGENCE_AVAILABLE = False
    TestIntelligence = None  # type: ignore
    create_test_intelligence = None  # type: ignore
    TestRunner = None  # type: ignore
    create_test_runner = None  # type: ignore
    ErrorAnalyzer = None  # type: ignore
    create_error_analyzer = None  # type: ignore
    Debugger = None  # type: ignore
    create_debugger = None  # type: ignore
from agent.session_commands import (
    list_sessions as list_sessions_cmd,
    show_session as show_session_cmd,
    delete_session as delete_session_cmd,
    continue_last_session,
    export_session as export_session_cmd,
)
from agent.test_runner import TestRunner
from agent.syntax_checker import SyntaxChecker, validate_python_files

app = typer.Typer(
    add_completion=False,
    help="Terminal-first offline code agent.",
    invoke_without_command=True,  # Allow running without explicit command
)
console = Console()


class PermissionManager:
    def __init__(self):
        self.always_allow = False

    def _select(self, message: str, options: list[str], default_index: int) -> int:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            text = Text()
            text.append(message, style="bold white")
            text.append("\n\n")
            for idx, opt in enumerate(options, start=1):
                style = "bold cyan" if (idx - 1) == default_index else "white"
                text.append(f"{idx}. {opt}\n", style=style)
            console.print(
                Panel(text, title="[bold]DECISION REQUIRED[/bold]", border_style="yellow", expand=False)
            )
            choice = Prompt.ask(
                "Select an option",
                choices=[str(i) for i in range(1, len(options) + 1)],
                default=str(default_index + 1),
                show_choices=False,
            )
            return int(choice) - 1

        selected = default_index

        def _read_escape_sequence(timeout_s: float = 0.08) -> str:
            buf = ""
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                remaining = max(0.0, deadline - time.time())
                if not select.select([sys.stdin], [], [], remaining)[0]:
                    break
                buf += sys.stdin.read(1)
                if buf and buf[-1] in ("A", "B", "C", "D"):
                    break
            return buf

        def _panel() -> Panel:
            text = Text()
            text.append(message, style="bold white")
            text.append("\n\n")
            for idx, opt in enumerate(options, start=1):
                is_sel = (idx - 1) == selected
                prefix = "> " if is_sel else "  "
                style = "bold cyan" if is_sel else "white"
                text.append(f"{prefix}{idx}. {opt}\n", style=style)
            text.append("\n↑/↓ or j/k, Enter", style="dim")
            return Panel(text, title="[bold]DECISION REQUIRED[/bold]", border_style="yellow", expand=False)

        def _draw(panel: Panel) -> None:
            console.print(panel)
            sys.stdout.flush()

        def _redraw(panel: Panel) -> None:
            sys.stdout.write("\x1b[u")
            sys.stdout.write("\x1b[J")
            sys.stdout.flush()
            _draw(panel)

        sys.stdout.write("\x1b[s")
        sys.stdout.flush()
        _draw(_panel())

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    return selected
                if ch in ("j",):
                    if selected < len(options) - 1:
                        selected += 1
                        _redraw(_panel())
                    continue
                if ch in ("k",):
                    if selected > 0:
                        selected -= 1
                        _redraw(_panel())
                    continue
                if ch == "\x1b":
                    seq = _read_escape_sequence()
                    code = seq[-1] if seq else ""
                    if code == "A" and selected > 0:
                        selected -= 1
                        _redraw(_panel())
                        continue
                    if code == "B" and selected < len(options) - 1:
                        selected += 1
                        _redraw(_panel())
                        continue
                    continue
                if ch.isdigit():
                    num = int(ch)
                    if 1 <= num <= len(options):
                        selected = num - 1
                        _redraw(_panel())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def multi_select(self, message: str, options: list[str]) -> list[int]:
        if not options:
            return []

        if not sys.stdin.isatty() or not sys.stdout.isatty():
            text = Text()
            text.append(message, style="bold white")
            text.append("\n\n")
            for idx, opt in enumerate(options, start=1):
                text.append(f"{idx}. {opt}\n", style="white")
            text.append("\nEnter comma-separated numbers (e.g. 1,3), 'all', or empty to cancel", style="dim")
            console.print(
                Panel(text, title="[bold]DECISION REQUIRED[/bold]", border_style="yellow", expand=False)
            )
            raw = Prompt.ask("Select", default="", show_default=False)
            val = raw.strip().lower()
            if not val:
                return []
            if val == "all":
                return list(range(len(options)))
            indices: list[int] = []
            for part in val.split(","):
                part = part.strip()
                if not part.isdigit():
                    continue
                idx = int(part) - 1
                if 0 <= idx < len(options):
                    indices.append(idx)
            return sorted(set(indices))

        selected: set[int] = set()
        cursor = 0

        def _read_escape_sequence(timeout_s: float = 0.08) -> str:
            buf = ""
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                remaining = max(0.0, deadline - time.time())
                if not select.select([sys.stdin], [], [], remaining)[0]:
                    break
                buf += sys.stdin.read(1)
                if buf and buf[-1] in ("A", "B", "C", "D"):
                    break
            return buf

        def _panel() -> Panel:
            text = Text()
            text.append(message, style="bold white")
            text.append("\n\n")
            for idx, opt in enumerate(options):
                is_cur = idx == cursor
                prefix = "> " if is_cur else "  "
                box = "[x]" if idx in selected else "[ ]"
                style = "bold cyan" if is_cur else "white"
                text.append(f"{prefix}{box} {idx + 1}. {opt}\n", style=style)
            text.append("\nSpace: toggle • a: all • Enter • Esc", style="dim")
            return Panel(text, title="[bold]DECISION REQUIRED[/bold]", border_style="yellow", expand=False)

        def _draw(panel: Panel) -> None:
            console.print(panel)
            sys.stdout.flush()

        def _redraw(panel: Panel) -> None:
            sys.stdout.write("\x1b[u")
            sys.stdout.write("\x1b[J")
            sys.stdout.flush()
            _draw(panel)

        sys.stdout.write("\x1b[s")
        sys.stdout.flush()
        _draw(_panel())

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    return sorted(selected)
                if ch == "\x1b":
                    seq = _read_escape_sequence()
                    code = seq[-1] if seq else ""
                    if code == "A" and cursor > 0:
                        cursor -= 1
                        _redraw(_panel())
                        continue
                    if code == "B" and cursor < len(options) - 1:
                        cursor += 1
                        _redraw(_panel())
                        continue
                    return []
                if ch in ("j",) and cursor < len(options) - 1:
                    cursor += 1
                    _redraw(_panel())
                    continue
                if ch in ("k",) and cursor > 0:
                    cursor -= 1
                    _redraw(_panel())
                    continue
                if ch == " ":
                    if cursor in selected:
                        selected.remove(cursor)
                    else:
                        selected.add(cursor)
                    _redraw(_panel())
                    continue
                if ch in ("a", "A"):
                    if len(selected) == len(options):
                        selected.clear()
                    else:
                        selected = set(range(len(options)))
                    _redraw(_panel())
                    continue
                if ch.isdigit():
                    num = int(ch)
                    if 1 <= num <= len(options):
                        cursor = num - 1
                        _redraw(_panel())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask user for confirmation with session-wide allow option."""
        if self.always_allow:
            return True

        options = ["Yes", "Yes, and don't ask again this session", "No"]
        default_index = 0 if default else 2
        choice_idx = self._select(message, options, default_index)

        if choice_idx == 0:
            return True
        elif choice_idx == 1:
            self.always_allow = True
            console.print("[dim]Permission granted for this session.[/dim]")
            return True
        else:
            return False

permission_manager = PermissionManager()


def _log_event(log_path: Path, event_type: str, payload: dict) -> None:
    """Log an event to the JSONL log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat() + "Z",
        "type": event_type,
        "payload": payload,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _confirm(message: str) -> bool:
    """Ask user for confirmation."""
    return permission_manager.confirm(message)


def _run_post_check(cmd: str) -> dict:
    """Run a post-apply check command and return result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }
    except Exception as exc:  # pragma: no cover
        return {"cmd": cmd, "error": str(exc)}


def _summarize_diff(diff_text: str) -> dict:
    """Extract summary stats from a unified diff."""
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    path = ""
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            path = line[4:].strip().split("\t")[0]
            break
    return {"path": path, "added": added, "removed": removed}


def _elapsed(started_at: float) -> float:
    return time.time() - started_at


def _event(started_at: float, message: str) -> None:
    """Print a progress event."""
    # Use show_plan as proxy for verbose mode (defined in run() function scope)
    # For now, always show timing to avoid scope issues


def _decision(message: str) -> bool:
    return _confirm(message)


def _is_refactoring_task(task: str) -> bool:
    """Detect if task is a refactoring operation."""
    task_lower = task.lower()
    refactoring_keywords = [
        "rename",
        "extract",
        "inline",
        "move",
        "refactor",
    ]
    return any(keyword in task_lower for keyword in refactoring_keywords)


def _handle_refactoring_task(task: str, refactor_tool: Any, plan: Any) -> Optional[Any]:
    """Handle refactoring task automatically."""
    if not refactor_tool:
        return None
    
    task_lower = task.lower()
    
    # Detect rename operation
    rename_match = re.search(r"rename\s+(?:function|class|method)\s+(\w+)\s+to\s+(\w+)", task_lower)
    if rename_match:
        old_name = rename_match.group(1)
        new_name = rename_match.group(2)
        symbol_type = "function" if "function" in task_lower else "class" if "class" in task_lower else "function"
        return refactor_tool.rename_symbol(old_name, new_name, symbol_type)
    
    # Detect extract operation
    extract_match = re.search(r"extract\s+(?:code|function|method)", task_lower)
    if extract_match and plan.files_to_modify:
        # Would need line numbers from task - simplified for now
        # In production, would parse task more carefully
        pass
    
    return None


def _choose_apply_action() -> str:
    options = ["Apply all", "Apply selected", "Skip"]
    idx = permission_manager._select("How would you like to apply these diffs?", options, default_index=0)
    return ["apply_all", "apply_selected", "skip"][idx]


def _stream_text(prompt: str, *, model: str | None) -> tuple[str, float | None]:
    started = time.time()
    first_token_at: float | None = None
    parts: list[str] = []
    for chunk in ask_stream(prompt, model=model):
        if chunk and chunk.strip() and first_token_at is None:
            first_token_at = time.time()
        parts.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()
    if parts and (not parts[-1].endswith("\n")):
        sys.stdout.write("\n")
        sys.stdout.flush()
    text = "".join(parts)
    thought_seconds = (first_token_at - started) if first_token_at is not None else None
    return text, thought_seconds

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    task_parts: list[str] = typer.Argument(None, help="The task description (can be multiple words)"),
    files: list[str] = typer.Option(None, "--file", "-f", help="Target files to focus on"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    targets: list[str] = typer.Option(None, "--targets", "-t", help="Target substrings/paths to prioritize"),
    dirs: list[str] = typer.Option(None, "--dirs", help="Limit search to these directories"),
    instruction: str = typer.Option("", "--instruction", "-i", help="Explicit edit instruction"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip apply step (review only)"),
    context: list[str] = typer.Option(None, "--context", "-c", help="Extra context file paths"),
    lang: str = typer.Option("", "--lang", help="Language/framework hint"),
    max_files: int = typer.Option(5, "--max-files", help="Max context files to read"),
    context_glob: list[str] = typer.Option(None, "--context-glob", help="Glob patterns for context files"),
    plan_only: bool = typer.Option(False, "--plan-only", help="Only show plan, skip execute/apply"),
    model: str = typer.Option("", "--model", help="Model override for this run"),
    mode: str = typer.Option("auto", "--mode", help="Prompt mode: auto|bugfix|refactor|feature|general"),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Use mock LLM (for testing)"),
    selective_apply: bool = typer.Option(False, "--selective-apply", help="Confirm per-diff before applying"),
    post_check: str = typer.Option("", "--post-check", help="Command to run after apply (e.g., tests)"),
    run_pytest: bool = typer.Option(
        False, "--run-pytest/--no-run-pytest", help="Run pytest -q after apply", show_default=False
    ),
    run_ruff: bool = typer.Option(
        False, "--run-ruff/--no-run-ruff", help="Run ruff check after apply", show_default=False
    ),
    run_mypy: bool = typer.Option(
        False, "--run-mypy/--no-run-mypy", help="Run mypy after apply", show_default=False
    ),
    run_py_compile: bool = typer.Option(
        False, "--run-py-compile/--no-run-py-compile", help="Run python -m py_compile **/*.py after apply", show_default=False
    ),
    show_plan: bool = typer.Option(
        False, "--show-plan/--no-show-plan", help="Display planner JSON and technical details", show_default=False
    ),
    show_reasoning: bool = typer.Option(
        True, "--show-reasoning/--no-show-reasoning", help="Display LLM prompt/response excerpts", show_default=True
    ),
    preview_context: bool = typer.Option(
        True, "--preview-context/--no-preview-context", help="Show context snippets sent to LLM", show_default=True
    ),
    symbols: str = typer.Option("", "--symbols", help="Comma-separated symbols to prioritize in context"),
    write_patch: str = typer.Option("", "--write-patch", help="Path to write combined diffs without applying"),
    relaxed_validation: bool = typer.Option(False, "--relaxed-validation", help="Loosen diff validation limits"),
    mock_from: str = typer.Option("", "--mock-from", help="Replay LLM outputs from a jsonl file"),
    allow_outside_repo: bool = typer.Option(
        False,
        "--allow-outside-repo/--disallow-outside-repo",
        help="Allow patch application outside the repo root (use with care)",
        show_default=True,
    ),
    quick_review: bool = typer.Option(
        False,
        "--quick-review",
        help="Stop after generating diffs and show review without apply prompt",
        show_default=False,
    ),
    enhance: bool = typer.Option(
        False,
        "--enhance",
        help="Use LLM to enhance and structure the task before planning",
        show_default=False,
    ),
    iterative: bool = typer.Option(
        False,
        "--iterative",
        help="Enable iterative refinement mode (allows multiple refinement rounds)",
        show_default=False,
    ),
    session: str = typer.Option(
        None,
        "--session",
        help="Continue or start a named conversation session",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        help="Watch mode: monitor file changes and auto-update context",
    ),
    list_sessions: bool = typer.Option(
        False,
        "--list-sessions",
        help="List all available conversation sessions",
    ),
) -> None:
    """Main entry point - starts interactive session if no task provided."""
    # Handle version flag early
    if version:
        try:
            from core import __version__
            console.print(f"Local Code Agent v{__version__}")
        except ImportError:
            console.print("Local Code Agent v0.1.0")
        raise typer.Exit(code=0)
    
    # Handle list-sessions flag
    if list_sessions:
        conv_manager = get_conversation_manager()
        sessions = conv_manager.list_sessions()
        if sessions:
            console.print("[bold]Available sessions:[/bold]")
            for sess_id in sessions:
                console.print(f"  • {sess_id}")
        else:
            console.print("[dim]No sessions found.[/dim]")
        raise typer.Exit(code=0)
    
    # Normalize task_parts to always be a list
    if task_parts is None:
        task_parts = []
    
    # If no task provided and no subcommand invoked, start interactive session
    if ctx.invoked_subcommand is None:
        if not task_parts:
            # Start interactive session by calling run() with no task_parts
            run(
                task_parts=[],
                files=files,
                targets=targets,
                dirs=dirs,
                instruction=instruction,
                dry_run=dry_run,
                context=context,
                lang=lang,
                max_files=max_files,
                context_glob=context_glob,
                plan_only=plan_only,
                model=model,
                mode=mode,
                mock_llm=mock_llm,
                selective_apply=selective_apply,
                post_check=post_check,
                run_pytest=run_pytest,
                run_ruff=run_ruff,
                run_mypy=run_mypy,
                run_py_compile=run_py_compile,
                show_plan=show_plan,
                show_reasoning=show_reasoning,
                preview_context=preview_context,
                symbols=symbols,
                write_patch=write_patch,
                relaxed_validation=relaxed_validation,
                mock_from=mock_from,
                allow_outside_repo=allow_outside_repo,
                quick_review=quick_review,
                enhance=enhance,
                iterative=iterative,
                session=session,
                watch=watch,
            )
        else:
            # Run with provided task
            run(
                task_parts=task_parts,
                files=files,
                version=version,
                targets=targets,
                dirs=dirs,
                instruction=instruction,
                dry_run=dry_run,
                context=context,
                lang=lang,
                max_files=max_files,
                context_glob=context_glob,
                plan_only=plan_only,
                model=model,
                mode=mode,
                mock_llm=mock_llm,
                selective_apply=selective_apply,
                post_check=post_check,
                run_pytest=run_pytest,
                run_ruff=run_ruff,
                run_mypy=run_mypy,
                run_py_compile=run_py_compile,
                show_plan=show_plan,
                show_reasoning=show_reasoning,
                preview_context=preview_context,
                symbols=symbols,
                write_patch=write_patch,
                relaxed_validation=relaxed_validation,
                mock_from=mock_from,
                allow_outside_repo=allow_outside_repo,
                quick_review=quick_review,
                enhance=enhance,
                iterative=iterative,
            )


@app.command(name="run", context_settings={"ignore_unknown_options": True})
def run(
    task_parts: list[str] = typer.Argument(None, help="The task description (can be multiple words)"),
    files: list[str] = typer.Option(None, "--file", "-f", help="Target files to focus on"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    targets: list[str] = typer.Option(None, "--targets", "-t", help="Target substrings/paths to prioritize"),
    dirs: list[str] = typer.Option(None, "--dirs", help="Limit search to these directories"),
    instruction: str = typer.Option("", "--instruction", "-i", help="Explicit edit instruction"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip apply step (review only)"),
    context: list[str] = typer.Option(None, "--context", "-c", help="Extra context file paths"),
    lang: str = typer.Option("", "--lang", help="Language/framework hint"),
    max_files: int = typer.Option(5, "--max-files", help="Max context files to read"),
    context_glob: list[str] = typer.Option(None, "--context-glob", help="Glob patterns for context files"),
    plan_only: bool = typer.Option(False, "--plan-only", help="Only show plan, skip execute/apply"),
    model: str = typer.Option("", "--model", help="Model override for this run"),
    mode: str = typer.Option("auto", "--mode", help="Prompt mode: auto|bugfix|refactor|feature|general"),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Use mock LLM (for testing)"),
    selective_apply: bool = typer.Option(False, "--selective-apply", help="Confirm per-diff before applying"),
    post_check: str = typer.Option("", "--post-check", help="Command to run after apply (e.g., tests)"),
    run_pytest: bool = typer.Option(
        False, "--run-pytest/--no-run-pytest", help="Run pytest -q after apply", show_default=False
    ),
    run_ruff: bool = typer.Option(
        False, "--run-ruff/--no-run-ruff", help="Run ruff check after apply", show_default=False
    ),
    run_mypy: bool = typer.Option(
        False, "--run-mypy/--no-run-mypy", help="Run mypy after apply", show_default=False
    ),
    run_py_compile: bool = typer.Option(
        False, "--run-py-compile/--no-run-py-compile", help="Run python -m py_compile **/*.py after apply", show_default=False
    ),
    show_plan: bool = typer.Option(
        False, "--show-plan/--no-show-plan", help="Display planner JSON and technical details", show_default=False
    ),
    show_reasoning: bool = typer.Option(
        True, "--show-reasoning/--no-show-reasoning", help="Display LLM prompt/response excerpts", show_default=True
    ),
    preview_context: bool = typer.Option(
        True, "--preview-context/--no-preview-context", help="Show context snippets sent to LLM", show_default=True
    ),
    symbols: str = typer.Option("", "--symbols", help="Comma-separated symbols to prioritize in context"),
    write_patch: str = typer.Option("", "--write-patch", help="Path to write combined diffs without applying"),
    relaxed_validation: bool = typer.Option(False, "--relaxed-validation", help="Loosen diff validation limits"),
    mock_from: str = typer.Option("", "--mock-from", help="Replay LLM outputs from a jsonl file"),
    allow_outside_repo: bool = typer.Option(
        False,
        "--allow-outside-repo/--disallow-outside-repo",
        help="Allow patch application outside the repo root (use with care)",
        show_default=True,
    ),
    quick_review: bool = typer.Option(
        False,
        "--quick-review",
        help="Stop after generating diffs and show review without apply prompt",
        show_default=False,
    ),
    enhance: bool = typer.Option(
        False,
        "--enhance",
        help="Use LLM to enhance and structure the task before planning",
        show_default=False,
    ),
    iterative: bool = typer.Option(
        False,
        "--iterative",
        help="Enable iterative refinement mode (allows multiple refinement rounds)",
        show_default=False,
    ),
    session: str = typer.Option(
        None,
        "--session",
        help="Continue or start a named conversation session",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        help="Watch mode: monitor file changes and auto-update context",
    ),
    auto_test: bool = typer.Option(
        False,
        "--auto-test",
        help="Automatically generate and run tests for generated code",
    ),
    debug_error: str = typer.Option(
        "",
        "--debug-error",
        help="Analyze and debug an error message or traceback",
    ),
) -> None:
    """
    Run the agent workflow: scan -> plan -> confirm -> execute -> confirm -> apply -> review.
    """
    def _coerce_option(value: Any, default: Any) -> Any:
        return default if isinstance(value, OptionInfo) else value

    # Normalize OptionInfo defaults when run() is invoked programmatically
    debug_error = _coerce_option(debug_error, "")
    auto_test = _coerce_option(auto_test, False)
    version = _coerce_option(version, False)
    dry_run = _coerce_option(dry_run, False)
    plan_only = _coerce_option(plan_only, False)
    mock_llm = _coerce_option(mock_llm, False)
    selective_apply = _coerce_option(selective_apply, False)
    run_pytest = _coerce_option(run_pytest, False)
    run_ruff = _coerce_option(run_ruff, False)
    run_mypy = _coerce_option(run_mypy, False)
    run_py_compile = _coerce_option(run_py_compile, False)
    show_plan = _coerce_option(show_plan, False)
    show_reasoning = _coerce_option(show_reasoning, True)
    preview_context = _coerce_option(preview_context, True)
    quick_review = _coerce_option(quick_review, False)
    enhance = _coerce_option(enhance, False)
    iterative = _coerce_option(iterative, False)
    allow_outside_repo = _coerce_option(allow_outside_repo, False)
    watch = _coerce_option(watch, False)
    session = _coerce_option(session, None)
    files = _coerce_option(files, None)
    targets = _coerce_option(targets, None)
    dirs = _coerce_option(dirs, None)
    context = _coerce_option(context, None)
    context_glob = _coerce_option(context_glob, None)
    symbols = _coerce_option(symbols, "")
    instruction = _coerce_option(instruction, "")
    lang = _coerce_option(lang, "")
    model = _coerce_option(model, "")
    mode = _coerce_option(mode, "auto")
    write_patch = _coerce_option(write_patch, "")
    relaxed_validation = _coerce_option(relaxed_validation, False)
    allow_outside_repo = bool(allow_outside_repo)

    # Input validation
    repo_root = str(config.repo_root)
    
    # Handle debug-error mode
    if debug_error:
        if TEST_INTELLIGENCE_AVAILABLE and create_error_analyzer:
            try:
                analyzer = create_error_analyzer(Path(repo_root))
                explanation = analyzer.explain_error(debug_error)
                console.print(Panel(Markdown(explanation), title="[bold]Error Analysis[/bold]", border_style="red"))
                
                # Generate debug suggestions
                if create_debugger:
                    debugger = create_debugger(Path(repo_root))
                    # Extract location from error if possible
                    location_match = re.search(r'File\s+"([^"]+)",\s+line\s+(\d+)', debug_error)
                    if location_match:
                        file_path = location_match.group(1)
                        line_num = location_match.group(2)
                        location = f"{file_path}:{line_num}"
                        suggestions = debugger.suggest_debug_points(location, "Error")
                        if suggestions:
                            console.print("\n[bold]Debug Suggestions:[/bold]")
                            for suggestion in suggestions[:3]:
                                console.print(f"  • {suggestion.suggestion_type}: {suggestion.code}")
                                console.print(f"    {suggestion.explanation}")
            except Exception as e:
                console.print(f"[red]Error analyzing: {e}[/red]")
        else:
            console.print("[yellow]Error analysis not available. Install dependencies.[/yellow]")
        return
    repo_path = Path(repo_root)
    if not repo_path.exists() or not repo_path.is_dir():
        console.print(
            f"[red]Error: Repository root '{repo_root}' does not exist or is not a directory.[/red]\n"
            f"[yellow]Hint: Set LCA_REPO_ROOT environment variable or run from within a repository.[/yellow]"
        )
        raise typer.Exit(code=1)

    # Validate model if provided
    if model:
        model_name = model.strip()
        if not model_name:
            console.print("[red]Error: Model name cannot be empty.[/red]")
            raise typer.Exit(code=1)
        # Note: We don't check if model exists here as it requires an LLM call
        # The LLM module will handle model validation with better error messages

    # Validate file paths if provided
    if files:
        invalid_files = []
        for file_path in files:
            full_path = repo_path / file_path
            if not full_path.exists():
                invalid_files.append(file_path)
        if invalid_files:
            console.print(
                f"[red]Error: The following files do not exist: {', '.join(invalid_files)}[/red]\n"
                f"[yellow]Hint: Check file paths relative to repository root: {repo_root}[/yellow]"
            )
            raise typer.Exit(code=1)

    # Validate context files if provided
    if context:
        invalid_context = []
        for ctx_path in context:
            full_path = repo_path / ctx_path
            if not full_path.exists():
                invalid_context.append(ctx_path)
        if invalid_context:
            console.print(
                f"[yellow]Warning: Some context files do not exist: {', '.join(invalid_context)}[/yellow]\n"
                f"[dim]Continuing without these files...[/dim]"
            )

    # Validate max_files
    if max_files < 1:
        console.print("[red]Error: --max-files must be at least 1.[/red]")
        raise typer.Exit(code=1)
    if max_files > 50:
        console.print(
            f"[yellow]Warning: --max-files={max_files} is very high and may cause performance issues.[/yellow]\n"
            f"[dim]Consider using a smaller value (default: 5)[/dim]"
        )

    # Validate mode
    valid_modes = {"auto", "bugfix", "refactor", "feature", "general"}
    if mode not in valid_modes:
        console.print(
            f"[red]Error: Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}[/red]"
        )
        raise typer.Exit(code=1)

    # Validate mock_from file if provided
    if mock_from:
        mock_path = Path(mock_from)
        if not mock_path.exists():
            console.print(
                f"[red]Error: Mock file '{mock_from}' does not exist.[/red]\n"
                f"[yellow]Hint: Create a JSONL file with one LLM response per line.[/yellow]"
            )
            raise typer.Exit(code=1)

    # Validate write_patch path if provided
    if write_patch:
        patch_path = Path(write_patch)
        if patch_path.exists() and not patch_path.is_file():
            console.print(
                f"[red]Error: Patch output path '{write_patch}' exists but is not a file.[/red]"
            )
            raise typer.Exit(code=1)
        # Create parent directory if it doesn't exist
        patch_path.parent.mkdir(parents=True, exist_ok=True)

    interactive_session = not task_parts
    initial_task = " ".join(task_parts) if task_parts else ""
    
    # Initialize conversation manager and handle session
    conv_manager = get_conversation_manager()
    if session:
        session_id = conv_manager.start_session(session)
        console.print(f"[dim]Using session: {session_id}[/dim]")
        # Show session history if exists
        context = conv_manager.get_context()
        if context["has_context"]:
            console.print(f"[dim]Previous tasks in this session: {len(context['previous_tasks'])}[/dim]")
    elif interactive_session:
        if not conv_manager.current_session_id:
            conv_manager.start_session()
    
    # If interactive session, show welcome message
    if interactive_session:
        console.print("\n[bold cyan]🤖 Local Code Agent[/bold cyan]")
        console.print("[dim]Type your task in plain English. Type 'exit' or 'quit' to exit.[/dim]\n")

    def _run_one(task: str) -> None:
        run_started_at = time.time()
        log_path = Path(repo_root) / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        # Conversation manager already initialized above

        # Check if this is a follow-up and expand context
        if conv_manager.is_follow_up(task):
            task = conv_manager.expand_follow_up(task)
            if not show_plan:
                console.print("[dim]Detected follow-up question, using previous context...[/dim]")

        # Classify task early to determine approach
        task_type, confidence = classify_task(task)
        is_conversational = is_conversational_query(task)
        needs_modification = should_modify_files(task) if not is_conversational else False

        # Cleaner task display
        if not show_plan and is_conversational:
            # For questions, show a cleaner prompt
            console.print(f"\n[bold blue]📚[/bold blue] [bold]{task}[/bold]\n")
        else:
            console.print(Panel(f"[bold blue]Task[/bold blue]: {task}", border_style="blue"))

        if enhance:
            from agent.prompt_engineer import stream_enhanced_task

            enhanced_task = ""
            console.print("[dim]Enhancing task...[/dim]")
            with Live(Markdown(""), refresh_per_second=10, transient=True) as live:
                for chunk in stream_enhanced_task(task, model=model):
                    enhanced_task += chunk
                    live.update(Markdown(enhanced_task))
            console.print(Panel(Markdown(enhanced_task), title="[bold]Enhanced Task[/bold]", border_style="magenta"))
            task = enhanced_task
            # Re-classify after enhancement
            task_type, confidence = classify_task(task)
            is_conversational = is_conversational_query(task)
            needs_modification = should_modify_files(task) if not is_conversational else False
            _log_event(log_path, "enhanced_task", {"original": task, "enhanced": enhanced_task})

        if not show_plan and is_conversational:
            # Hide technical details for cleaner output
            _event(run_started_at, "Understanding your codebase...")
        else:
            _event(run_started_at, "Scanning...")
        
        repo_files = scan_repo(repo_root)
        
        if show_plan:
            _event(run_started_at, f"Discovered {len(repo_files)} files")

        # Use pattern matching for better file selection (after repo_files is available)
        pattern_matcher = PatternMatcher()
        suggested_files = []
        if not is_conversational:
            try:
                suggested_files = pattern_matcher.suggest_files(task, task_type.value, repo_files)
                if suggested_files and not show_plan:
                    console.print(f"[dim]Found {len(suggested_files)} similar past tasks[/dim]")
            except Exception:
                # If pattern matching fails, continue without it
                pass

        if not is_conversational:
            if not show_plan:
                _event(run_started_at, "Planning changes...")
            else:
                _event(run_started_at, "Planning...", status="info")
            plan = build_plan(task, repo_files, targets=targets, dirs=dirs, repo_root=repo_root)
            
            # Enhance plan with pattern-based suggestions
            if suggested_files and not plan.files_to_read:
                # Add suggested files from patterns
                plan.files_to_read = suggested_files[:3]  # Top 3 suggestions
                if not plan.files_to_modify:
                    plan.files_to_modify = suggested_files[:3]
            
            if files:
                plan.files_to_read = files
                plan.files_to_modify = files
            if show_plan:
                _event(run_started_at, "Planning done")
            else:
                _event(run_started_at, "Ready")

            plan_json = plan_to_json(plan)
            _log_event(log_path, "plan", json.loads(plan_json))
        else:
            # For conversational queries, create a minimal plan
            plan = build_plan(task, repo_files, targets=targets, dirs=dirs, repo_root=repo_root)
            plan.files_to_modify = []  # Never modify for questions
            plan_json = plan_to_json(plan)
            _log_event(log_path, "plan", json.loads(plan_json))

        if show_plan:
            console.print(Panel(Syntax(plan_json, "json", word_wrap=True), title="[bold]Plan JSON[/bold]", border_style="green"))

        # Only show technical details if verbose mode
        if show_plan:
            if plan.files_to_read:
                _event(run_started_at, f"Will read {', '.join(plan.files_to_read)}")
            if plan.files_to_modify:
                _event(run_started_at, f"Will modify {', '.join(plan.files_to_modify)}")
            if plan.steps:
                _event(run_started_at, f"Next: {plan.steps[0]}")

        if context_glob:
            console.print(f"[dim]Context globs: {context_glob}[/dim]")
        if instruction:
            console.print(f"[dim]Instruction override: {instruction}[/dim]")
            _log_event(log_path, "instruction", {"instruction": instruction})

        if plan_only:
            console.print("[yellow]Plan-only mode completed.[/yellow]")
            _log_event(log_path, "plan_only", {"plan_only": True})
            return

        if is_conversational:
            # For conversational queries, skip confirmation and provide direct answer
            # Gather context intelligently
            if not plan.files_to_read:
                # More aggressive search for documentation files
                doc_patterns = [
                    "README.md", "README.txt", "README.rst", "README",
                    "readme.md", "readme.txt", "readme",
                    "docs/README.md", "docs/readme.md",
                    "ARCHITECTURE.md", "ARCHITECTURE.txt",
                    "PROJECT_STATUS.md", "DEVELOPMENT_ROADMAP.md",
                    "CONTRIBUTING.md", "CONTRIBUTING.txt",
                    "ABOUT.md", "DESCRIPTION.md",
                ]
                
                # Search for docs in repo_files (exact match)
                found_docs = [f for f in doc_patterns if f in repo_files]
                
                # Also search for files starting with these patterns
                if not found_docs:
                    for pattern in doc_patterns:
                        pattern_base = pattern.split("/")[-1].replace(".md", "").replace(".txt", "").lower()
                        matches = [f for f in repo_files if f.lower().endswith(pattern_base) or 
                                  f.lower().endswith(pattern_base + ".md") or 
                                  f.lower().endswith(pattern_base + ".txt")]
                        if matches:
                            found_docs.extend(matches[:2])
                            break
                
                # Search in docs/ directory
                if not found_docs:
                    docs_dir_files = [f for f in repo_files if f.startswith("docs/") and 
                                     (f.endswith(".md") or f.endswith(".txt") or "readme" in f.lower())]
                    if docs_dir_files:
                        found_docs.extend(docs_dir_files[:3])
                
                if found_docs:
                    plan.files_to_read = list(set(found_docs))[:5]  # Top 5 unique docs
                else:
                    # Fallback: search for any markdown files in root
                    md_files = [f for f in repo_files if f.endswith(".md") and "/" not in f]
                    if md_files:
                        plan.files_to_read = md_files[:3]
                    else:
                        # Last resort: common config files that might have project info
                        defaults = ["pyproject.toml", "package.json", "setup.py", "setup.cfg", 
                                   "requirements.txt", "Cargo.toml", "go.mod", "pom.xml"]
                        found_defaults = [f for f in defaults if f in repo_files]
                        if found_defaults:
                            plan.files_to_read = found_defaults[:3]
                        else:
                            # If still nothing, try to use planner's suggestions
                            plan = build_plan(task, repo_files, targets=targets, dirs=dirs, repo_root=repo_root)
                            if plan.files_to_read:
                                plan.files_to_read = plan.files_to_read[:5]

            def _normalize_rel(p: str) -> str:
                pp = Path(p)
                if pp.is_absolute():
                    try:
                        return str(pp.relative_to(repo_root))
                    except ValueError:
                        return pp.name
                return pp.as_posix()

            if plan.files_to_read:
                plan.files_to_read = [_normalize_rel(p) for p in plan.files_to_read]

            # Build context with semantic search if symbols are mentioned
            from agent.executor import build_context
            
            # For "what does this project do" questions, be more aggressive
            is_project_question = any(phrase in task.lower() for phrase in [
                "what does this project", "what does the project", "what is this project",
                "what is the project", "describe this project", "explain this project"
            ])
            
            task_words = task.lower().split()
            # Extract potential symbols/keywords from question
            keywords = [w for w in task_words if len(w) > 3 and w not in ["what", "does", "this", "project", "how", "why", "when", "where"]]
            
            if not show_plan:
                _event(run_started_at, "Gathering context...")
            snippets = []
            
            # Try to read files even if plan doesn't have them
            files_to_read = plan.files_to_read or []
            
            # If no files found and it's a project question, try harder
            if not files_to_read and is_project_question:
                # Look for any README-like files
                for f in repo_files:
                    if "readme" in f.lower() or f.endswith(".md"):
                        files_to_read.append(f)
                        if len(files_to_read) >= 3:
                            break
            
            if files_to_read:
                try:
                    snippets = build_context(
                        Path(repo_root), 
                        files_to_read, 
                        max_context_files=max(max_files, 10),  # Read more files for questions
                        symbols=keywords if keywords else None,
                        include_related=False  # Don't expand for questions
                    )
                except Exception as e:
                    if not show_plan:
                        console.print(f"[dim]Warning: Could not read some files: {e}[/dim]")
                    # Try reading files directly as fallback
                    snippets = []
                    for file_path in files_to_read[:5]:
                        try:
                            full_path = Path(repo_root) / file_path
                            if full_path.exists():
                                content = full_path.read_text(encoding="utf-8", errors="replace")
                                # Limit content size
                                content = content[:5000]  # First 5000 chars
                                snippets.append({
                                    "path": file_path,
                                    "snippet": content
                                })
                        except Exception:
                            continue
            
            # Build a clean prompt for answering
            context_parts = []
            if snippets:
                for snippet in snippets[:10]:  # Limit to top 10 snippets for questions
                    snippet_content = snippet.get('snippet', '')
                    if snippet_content:
                        context_parts.append(f"**{snippet.get('path', 'unknown')}**\n```\n{snippet_content[:2000]}\n```")
            
            if not context_parts:
                # If still no context, try to read main files directly
                main_files = ["README.md", "README", "readme.md", "readme.txt"]
                for main_file in main_files:
                    try:
                        full_path = Path(repo_root) / main_file
                        if full_path.exists():
                            content = full_path.read_text(encoding="utf-8", errors="replace")[:5000]
                            context_parts.append(f"**{main_file}**\n```\n{content}\n```")
                            break
                    except Exception:
                        continue
            
            context_text = "\n\n".join(context_parts) if context_parts else (
                "No documentation files found in the project. "
                "The project structure suggests this is a codebase, but no README or documentation files were located."
            )
            
            # Clean, professional prompt
            if is_project_question:
                prompt = (
                    f"You are a helpful coding assistant. Answer this question about the codebase:\n\n"
                    f"**Question:** {task}\n\n"
                    f"**Context from codebase:**\n{context_text}\n\n"
                    f"Based on the provided context, explain what this project does. "
                    f"Focus on:\n"
                    f"1. The main purpose and functionality\n"
                    f"2. Key features and capabilities\n"
                    f"3. Technologies or frameworks used (if evident from file names or content)\n"
                    f"4. Project structure (if visible from file organization)\n\n"
                    f"If the context is limited, infer what you can from file names, structure, and any available content. "
                    f"Use markdown formatting for readability. Be specific and helpful."
                )
            else:
                prompt = (
                    f"You are a helpful coding assistant. Answer this question about the codebase:\n\n"
                    f"**Question:** {task}\n\n"
                    f"**Context from codebase:**\n{context_text}\n\n"
                    f"Provide a clear, concise answer based on the provided context. "
                    f"If the context is limited, do your best to answer based on what's available. "
                    f"Use markdown formatting for readability. "
                    f"If you reference specific files, mention them by name."
                )

            if not show_plan:
                _event(run_started_at, "Generating answer...")
            else:
                _event(run_started_at, "Generating...")
            
            answer_text, thought = _stream_text(prompt, model=model or None)
            if not answer_text.strip():
                # Fallback
                answer_text = ask(prompt, model=model or None)
            
            # Display answer cleanly
            console.print("\n")
            console.print(Panel(Markdown(answer_text), title="[bold]Answer[/bold]", border_style="blue", padding=(1, 2)))
            
            # Show sources if available
            if snippets:
                sources = [s['path'] for s in snippets[:3]]
                console.print(f"\n[dim]Sources: {', '.join(sources)}[/dim]")
            
            if thought is not None and show_plan:
                _event(run_started_at, f"Thought {int(round(thought))}s")
            
            # Record conversational turn
            try:
                conv_manager.add_turn(
                    task=task,
                    task_type=task_type.value,
                    response=answer_text,
                    files=[s['path'] for s in snippets] if snippets else []
                )
            except Exception:
                pass
            
            _log_event(log_path, "conversational_answer", {
                "question": task,
                "answer": answer_text,
                "sources": [s['path'] for s in snippets] if snippets else []
            })
            console.print(f"\n[dim]Log saved to {log_path}[/dim]")
            return

        context_files = context or plan.files_to_read
        if context_glob:
            extra = []
            for pattern in context_glob:
                extra.extend([str(p.relative_to(repo_root)) for p in Path(repo_root).glob(pattern)])
            context_files = list(dict.fromkeys(context_files + extra))
            console.print(f"[dim]Context glob matched {len(extra)} files[/dim]")

        if not is_conversational:
            if not _decision("Proceed to generate changes?"):
                console.print("[yellow]Aborted.[/yellow]")
                raise typer.Exit(code=0)

        exec_instruction = instruction if instruction else ""

        llm_stub_fn: Callable[[str], str] | None = None
        if mock_llm:
            def _stub_mock(_: str) -> str:
                target = plan.files_to_modify[0] if plan.files_to_modify else "mock.txt"
                return f"""--- {target}
+++ {target}
@@ -1 +1 @@
-placeholder
+mock-change
"""

            llm_stub_fn = _stub_mock
        elif mock_from:
            lines = Path(mock_from).read_text(encoding="utf-8").splitlines()

            def _stub_replay(_: str, _lines=iter(lines)) -> str:
                try:
                    return next(_lines)
                except StopIteration:
                    return ""

            llm_stub_fn = _stub_replay

        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

        def _wrapped_llm(prompt_text: str) -> str:
            if llm_stub_fn is not None:
                return llm_stub_fn(prompt_text)
            if show_reasoning and sys.stdout.isatty():
                _event(run_started_at, "Generating diffs...")
                text, thought = _stream_text(prompt_text, model=model or None)
                if thought is not None:
                    _event(run_started_at, f"Thought {int(round(thought))}s")
                if not text.strip():
                    return ask(prompt_text, model=model or None)
                return text
            return ask(prompt_text, model=model or None)

        # Check if task needs decomposition
        decomposed: Optional[DecomposedTask] = None
        if not is_conversational and not dry_run:
            try:
                decomposed = decompose_task(task, repo_files, use_llm=True, llm_fn=_wrapped_llm if not mock_llm else None)
                if len(decomposed.subtasks) > 1:
                    if not show_plan:
                        console.print(f"[cyan]📋 Task decomposed into {len(decomposed.subtasks)} subtasks[/cyan]")
                    _log_event(log_path, "decomposition", {
                        "subtasks": [{"id": st.id, "description": st.description, "dependencies": st.dependencies} for st in decomposed.subtasks],
                        "execution_order": decomposed.execution_order,
                    })
            except Exception:
                # If decomposition fails, continue with single-step execution
                decomposed = None

        # Execute: multi-step if decomposed, single-step otherwise
        if decomposed and len(decomposed.subtasks) > 1:
            # Multi-step execution
            _event(run_started_at, f"Executing {len(decomposed.subtasks)} subtasks...", status="info")
            all_diffs = []
            all_notes = []
            min_confidence = 1.0
            
            for subtask_id in decomposed.execution_order:
                subtask = next(st for st in decomposed.subtasks if st.id == subtask_id)
                if subtask.status != SubtaskStatus.PENDING:
                    continue
                
                subtask.status = SubtaskStatus.IN_PROGRESS
                if not show_plan:
                    console.print(f"[dim]  Subtask {subtask_id}/{len(decomposed.subtasks)}: {subtask.description[:50]}...[/dim]")
                
                try:
                    # Execute subtask
                    subtask_result = execute(
                        subtask.description,
                        subtask.files_to_read or plan.files_to_read,
                        subtask.files_to_modify or plan.files_to_modify,
                        repo_root,
                        instruction=exec_instruction or None,
                        context_paths=context_files,
                        max_context_files=max_files,
                        language=lang or None,
                        model=model or None,
                        mode=None if mode == "auto" else mode,
                        llm_fn=_wrapped_llm,
                        symbols=symbol_list or None,
                        relaxed_validation=relaxed_validation,
                        enable_llm=True,
                    )
                    
                    if subtask_result.diffs:
                        # Validate before applying
                        for diff_entry in subtask_result.diffs:
                            diff_path = diff_entry.get("path", "")
                            diff_text = diff_entry.get("diff", "")
                            
                            if diff_path:
                                validation = validate_before_apply(diff_text, diff_path, repo_root, auto_correct=True)
                                if not validation.valid and validation.corrected_diff:
                                    # Use corrected diff
                                    diff_entry["diff"] = validation.corrected_diff
                                    if not show_plan:
                                        console.print(f"[dim]    Auto-corrected diff for {diff_path}[/dim]")
                                elif not validation.valid:
                                    subtask.status = SubtaskStatus.FAILED
                                    subtask.error = "; ".join(validation.errors)
                                    if not show_plan:
                                        console.print(f"[yellow]    Validation failed for {diff_path}: {validation.errors[0]}[/yellow]")
                                    continue
                            
                            all_diffs.append(diff_entry)
                    
                    all_notes.extend(subtask_result.notes)
                    min_confidence = min(min_confidence, subtask_result.confidence)
                    subtask.result = json.loads(executor_to_json(subtask_result))
                    subtask.status = SubtaskStatus.COMPLETED
                    
                except Exception as e:
                    subtask.status = SubtaskStatus.FAILED
                    subtask.error = str(e)
                    if not show_plan:
                        console.print(f"[red]    Subtask {subtask_id} failed: {e}[/red]")
                    
                    # Decide whether to continue or abort
                    # For now, continue with other subtasks
                    continue
            
            # Combine results
            from agent.executor import ExecutorResult
            result = ExecutorResult(
                diffs=all_diffs,
                notes=all_notes,
                confidence=min_confidence,
                context_preview=None,  # Multi-step doesn't preserve context preview
            )
        else:
            # Single-step execution
            _event(run_started_at, "Executing...", status="info")
            # Check if this is a refactoring task and use refactoring tool automatically
            is_refactoring_task = _is_refactoring_task(task)
            if is_refactoring_task and REFACTORING_AVAILABLE and create_refactoring_tool:
                try:
                    refactor_tool = create_refactoring_tool(Path(repo_root))
                    refactor_plan = _handle_refactoring_task(task, refactor_tool, plan)
                    if refactor_plan and refactor_plan.changes:
                        # Convert refactoring plan to executor result format
                        diffs = []
                        for change in refactor_plan.changes:
                            from tools.diff import unified_diff
                            diff_text = unified_diff(
                                change["old_code"],
                                change["new_code"],
                                change["file"],
                                change["file"],
                            )
                            diffs.append({"path": change["file"], "diff": diff_text})
                        
                        # Show refactoring plan
                        if refactor_plan.risks:
                            console.print(f"[yellow]Refactoring risks: {', '.join(refactor_plan.risks)}[/yellow]")
                        
                        result = ExecutorResult(
                            diffs=diffs,
                            notes=[f"Refactoring: {refactor_plan.operation} {refactor_plan.target_symbol}"],
                            confidence=0.8,
                        )
                    else:
                        # Fall back to normal execution
                        result = execute(
                            task,
                            plan.files_to_read,
                            plan.files_to_modify,
                            repo_root,
                            instruction=exec_instruction or None,
                            context_paths=context_files,
                            max_context_files=max_files,
                            language=lang or None,
                            model=model or None,
                            mode=None if mode == "auto" else mode,
                            llm_fn=_wrapped_llm,
                            symbols=symbol_list or None,
                            relaxed_validation=relaxed_validation,
                            enable_llm=True,
                        )
                except Exception:
                    # If refactoring fails, fall back to normal execution
                    result = execute(
                        task,
                        plan.files_to_read,
                        plan.files_to_modify,
                        repo_root,
                        instruction=exec_instruction or None,
                        context_paths=context_files,
                        max_context_files=max_files,
                        language=lang or None,
                        model=model or None,
                        mode=None if mode == "auto" else mode,
                        llm_fn=_wrapped_llm,
                        symbols=symbol_list or None,
                        relaxed_validation=relaxed_validation,
                        enable_llm=True,
                    )
 
            # Validate diffs before showing them
            if result.diffs:
                validated_diffs = []
                for diff_entry in result.diffs:
                    diff_path = diff_entry.get("path", "")
                    diff_text = diff_entry.get("diff", "")
                    
                    if diff_path:
                        validation = validate_before_apply(diff_text, diff_path, repo_root, auto_correct=True)
                        if validation.corrected_diff:
                            diff_entry["diff"] = validation.corrected_diff
                        if not validation.valid:
                            # Still include but mark as having warnings
                            diff_entry["validation_warnings"] = validation.warnings
                    
                    validated_diffs.append(diff_entry)
                result.diffs = validated_diffs
        
        _log_event(log_path, "execute", json.loads(executor_to_json(result)))

        if preview_context and result.context_preview:
            used = [c["path"] for c in result.context_preview]
            _event(run_started_at, f"Read {', '.join(used)}", status="info")

        # Show validation warnings if any
        if result.diffs and not show_plan:
            for diff_entry in result.diffs:
                if "validation_warnings" in diff_entry:
                    warnings = diff_entry["validation_warnings"]
                    if warnings:
                        console.print(f"[yellow]⚠ Validation warnings for {diff_entry.get('path', 'unknown')}: {warnings[0]}[/yellow]")

        if not result.diffs:
            # For conversational queries, provide an answer even if no diffs
            if is_conversational:
                # Try to answer the question using available context
                if result.context_preview:
                    context_text = "\n\n".join([f"**{c['path']}**\n{c['snippet'][:500]}..." for c in result.context_preview[:3]])
                    answer_prompt = (
                        f"Based on the following codebase context, answer this question:\n\n"
                        f"Question: {task}\n\n"
                        f"Context:\n{context_text}\n\n"
                        f"Provide a clear, helpful answer. If the context doesn't contain enough information, "
                        f"say what you can infer from what's available."
                    )
                    try:
                        answer = ask(answer_prompt, model=model or None)
                        console.print("\n")
                        console.print(Panel(Markdown(answer), title="[bold]Answer[/bold]", border_style="blue"))
                        _log_event(log_path, "conversational_answer", {"question": task, "answer": answer})
                    except Exception as exc:
                        console.print(f"[yellow]Could not generate answer: {exc}[/yellow]")
                else:
                    console.print("[yellow]No context available to answer the question.[/yellow]")
            else:
                console.print("[yellow]No changes were generated.[/yellow]")
                if result.notes:
                    console.print("[dim]Notes:[/dim]")
                    for note in result.notes:
                        console.print(f"  [dim]• {note}[/dim]")
            
            review = review_diffs(task, result.diffs)
            if show_plan:
                console.print(Panel("[bold]Review[/bold]\n" + reviewer_to_json(review)))
            _log_event(log_path, "review", json.loads(reviewer_to_json(review)))
            console.print(f"\n[dim]Log saved to {log_path}[/dim]")
            return

        if quick_review:
            console.print("[blue]Quick review enabled: skipping apply prompts.[/blue]")
            review = review_diffs(task, result.diffs)
            console.print(Panel("[bold]Review[/bold]\n" + reviewer_to_json(review)))
            _log_event(log_path, "review", json.loads(reviewer_to_json(review)))
            console.print(f"[cyan]Log saved to {log_path}[/cyan]")
            return

        if write_patch:
            patch_content = "\n\n".join(d.get("diff", "") for d in result.diffs if d.get("diff"))
            Path(write_patch).write_text(patch_content, encoding="utf-8")
            console.print(f"[blue]Diffs written to {write_patch}[/blue]")

        summaries = [_summarize_diff(d.get("diff", "")) for d in result.diffs]
        console.print(
            Panel(
                "\n".join(
                    f"{s['path'] or '(unknown)'}  [green]+{s['added']}[/green] [red]-{s['removed']}[/red]"
                    for s in summaries
                ),
                title="[bold]Diff summaries[/bold]",
                border_style="dim",
            )
        )

        # Iterative refinement loop
        refinement_round = 0
        max_refinements = 3 if iterative else 0
        
        while refinement_round <= max_refinements:
            if refinement_round > 0:
                # Ask for refinement
                refinement_options = ["Apply as-is", "Refine changes", "Skip"]
                refine_idx = permission_manager._select(
                    "How would you like to proceed?",
                    refinement_options,
                    default_index=0
                )
                
                if refine_idx == 2:  # Skip
                    break
                elif refine_idx == 1:  # Refine
                    refinement_request = Prompt.ask(
                        "What changes would you like? (e.g., 'make it simpler', 'add error handling')",
                        default=""
                    )
                    if refinement_request:
                        # Re-execute with refinement
                        refined_task = f"{task} (refinement: {refinement_request})"
                        if not show_plan:
                            console.print(f"[dim]Refining based on: {refinement_request}[/dim]")
                        
                        # Re-execute with refinement context
                        refinement_result = execute(
                            refined_task,
                            plan.files_to_read,
                            plan.files_to_modify,
                            repo_root,
                            instruction=refinement_request,
                            context_paths=context_files,
                            max_context_files=max_files,
                            language=lang or None,
                            model=model or None,
                            mode=None if mode == "auto" else mode,
                            llm_fn=_wrapped_llm,
                            symbols=symbol_list or None,
                            relaxed_validation=relaxed_validation,
                            enable_llm=True,
                        )
                        
                        if refinement_result.diffs:
                            result = refinement_result
                            summaries = [_summarize_diff(d.get("diff", "")) for d in result.diffs]
                            console.print(
                                Panel(
                                    "\n".join(
                                        f"{s['path'] or '(unknown)'}  [green]+{s['added']}[/green] [red]-{s['removed']}[/red]"
                                        for s in summaries
                                    ),
                                    title="[bold]Refined diff summaries[/bold]",
                                    border_style="dim",
                                )
                            )
                            refinement_round += 1
                            continue
                        else:
                            console.print("[yellow]No changes generated from refinement.[/yellow]")
                            break
                    else:
                        break
            
            action = _choose_apply_action()
            break  # Exit refinement loop after first action choice
        if action == "skip":
            console.print("[yellow]Skipping apply.[/yellow]")
            review = review_diffs(task, result.diffs)
            if show_plan:
                console.print(Panel("[bold]Review[/bold]\n" + reviewer_to_json(review)))
            _log_event(log_path, "review", json.loads(reviewer_to_json(review)))
            console.print(f"[cyan]Log saved to {log_path}[/cyan]")
            return

        if dry_run:
            console.print("[yellow]Dry-run: skipping apply step.[/yellow]")
            review = review_diffs(task, result.diffs)
            if show_plan:
                console.print(Panel("[bold]Review[/bold]\n" + reviewer_to_json(review)))
            _log_event(log_path, "review", json.loads(reviewer_to_json(review)))
            console.print(f"[cyan]Log saved to {log_path}[/cyan]")
            return

        applied: list[str] = []
        failed: list[dict[str, str]] = []
        diffs_to_apply = result.diffs
        if action == "apply_selected" or selective_apply:
            options = []
            for s in summaries:
                path = s["path"] or "(unknown)"
                options.append(f"{path}  +{s['added']} -{s['removed']}")
            chosen = permission_manager.multi_select("Select diffs to apply", options)
            diffs_to_apply = [result.diffs[i] for i in chosen]
            if not diffs_to_apply:
                console.print("[yellow]No diffs selected; skipping apply.[/yellow]")
                review = review_diffs(task, result.diffs)
                if show_plan:
                    console.print(Panel("[bold]Review[/bold]\n" + reviewer_to_json(review)))
                _log_event(log_path, "review", json.loads(reviewer_to_json(review)))
                console.print(f"[cyan]Log saved to {log_path}[/cyan]")
                return

        # Apply patches with partial failure handling
        for diff_entry in diffs_to_apply:
            diff_text = diff_entry.get("diff", "")
            path = diff_entry.get("path", "unknown")
            if not diff_text.strip():
                failed.append({"path": path, "error": "Empty diff text"})
                continue
            try:
                applied_files = apply_patch(diff_text, allow_outside_repo=allow_outside_repo)
                applied.extend(applied_files)
                if applied_files:
                    _event(run_started_at, f"Applied patch to {path}", status="success")
            except PatchError as exc:
                error_msg = str(exc)
                failed.append({"path": path, "error": error_msg})
                console.print(
                    f"[red]✗ Failed to apply patch for {path}[/red]\n"
                    f"[dim]  Error: {error_msg}[/dim]\n"
                    f"[yellow]  Continuing with remaining patches...[/yellow]"
                )
                _log_event(log_path, "apply_error", {"path": path, "error": error_msg})

        # Summary of application results
        if applied:
            console.print(f"[green]✓ Successfully applied patches to {len(applied)} file(s):[/green]")
            for path in applied:
                console.print(f"  [green]  • {path}[/green]")
        else:
            console.print("[yellow]No patches were successfully applied.[/yellow]")

        if failed:
            console.print(f"[red]✗ Failed to apply {len(failed)} patch(es):[/red]")
            for failure in failed:
                console.print(f"  [red]  • {failure['path']}: {failure['error']}[/red]")
            console.print(
                "[yellow]Hint: Review the failed patches manually. "
                "You can use --write-patch to save diffs for manual application.[/yellow]"
            )

        _log_event(log_path, "apply", {"applied": applied, "failed": failed})

        # Record task in history for learning
        if not is_conversational:
            execution_time = _elapsed(run_started_at)
            success = len(applied) > 0
            try:
                learn_from_task(
                    task=task,
                    task_type=task_type.value,
                    files_read=plan.files_to_read,
                    files_modified=applied,
                    success=success,
                    execution_time=execution_time,
                    confidence=result.confidence,
                    notes=result.notes,
                    plan_rationale=plan.assumptions,
                )
            except Exception:
                # If learning fails, continue without it
                pass

        # Add to conversation context
        try:
            conv_manager.add_turn(
                task=task,
                task_type=task_type.value,
                files=plan.files_to_read + applied
            )
        except Exception:
            pass

        commands: list[str] = []
        if run_py_compile:
            commands.append("python -m compileall -q .")
        # Auto-test: Generate and run tests if enabled
        if auto_test and TEST_INTELLIGENCE_AVAILABLE and create_test_intelligence:
            try:
                if not show_plan:
                    console.print("[dim]Auto-test mode: Generating tests...[/dim]")
                
                test_intelligence = create_test_intelligence(Path(repo_root))
                
                # Generate tests for modified files
                for diff_entry in result.diffs:
                    file_path = diff_entry.get("path", "")
                    if file_path and file_path.endswith(".py"):
                        # Find functions in the diff
                        diff_text = diff_entry.get("diff", "")
                        # Extract function names from diff (simplified)
                        func_pattern = r"def\s+(\w+)\s*\("
                        functions = re.findall(func_pattern, diff_text)
                        
                        for func_name in functions[:3]:  # Limit to 3 functions
                            # Generate test
                            test_code = test_intelligence.generate_test(
                                function_name=func_name,
                                function_code=diff_text,
                            )
                            
                            # Find or create test file
                            source_file = Path(repo_root) / file_path
                            test_file = test_intelligence.find_test_file(source_file)
                            
                            if not test_file:
                                # Create test file
                                test_dir = Path(repo_root) / "tests"
                                test_dir.mkdir(exist_ok=True)
                                test_file = test_dir / f"test_{source_file.stem}.py"
                            
                            # Append test to file
                            if test_file.exists():
                                existing = test_file.read_text()
                                if func_name not in existing:
                                    test_file.write_text(existing + "\n\n" + test_code)
                            else:
                                test_file.write_text(test_code)
                            
                            if not show_plan:
                                console.print(f"[dim]Generated test for {func_name} in {test_file}[/dim]")
                
                # Run tests
                if create_test_runner:
                    test_runner = create_test_runner(Path(repo_root))
                    test_result = test_runner.run_tests(verbose=True)
                    
                    if test_result.failed > 0:
                        console.print(f"[yellow]Tests: {test_result.passed} passed, {test_result.failed} failed[/yellow]")
                    else:
                        console.print(f"[green]Tests: {test_result.passed} passed[/green]")
            except Exception as e:
                if not show_plan:
                    console.print(f"[yellow]Auto-test failed: {e}[/yellow]")
        
        if run_pytest:
            commands.append("pytest -q")
        if run_ruff:
            commands.append("ruff check .")
        if run_mypy:
            commands.append("mypy .")

        preset = post_check.strip()
        if preset:
            commands.append(preset)

        auto_added_default = False
        if not commands:
            if _decision("Run default post-check `python -m compileall -q .`?"):
                commands.append("python -m compileall -q .")
                auto_added_default = True

        if commands:
            if not auto_added_default and not _decision(f"Run {len(commands)} post-check commands?"):
                console.print("[yellow]Skipping post-checks.[/yellow]")
                commands = []

        if commands:
            console.print(f"[blue]Running post-checks ({len(commands)}): {', '.join(commands)}[/blue]")
            for cmd in commands:
                console.print(f"[blue]Post-check: {cmd}[/blue]")
                post_result = _run_post_check(cmd)
                status = "ok" if post_result.get("returncode", 1) == 0 else "fail"
                console.print(f"[blue]Status: {status}[/blue]")
                if post_result.get("stdout"):
                    console.print(f"[dim]stdout tail:\n{post_result['stdout']}[/dim]")
                if post_result.get("stderr"):
                    console.print(f"[red]stderr tail:\n{post_result['stderr']}[/red]")
                _log_event(log_path, "post_check", {"cmd": cmd, **post_result})

        review = review_diffs(task, result.diffs)
        if show_plan:
            console.print(Panel("[bold]Review[/bold]\n" + reviewer_to_json(review)))
        _log_event(log_path, "review", json.loads(reviewer_to_json(review)))

        # Record task in history for learning
        if not is_conversational:
            execution_time = _elapsed(run_started_at)
            success = len(applied) > 0 if 'applied' in locals() else False
            try:
                learn_from_task(
                    task=task,
                    task_type=task_type.value,
                    files_read=plan.files_to_read,
                    files_modified=applied if 'applied' in locals() else [],
                    success=success,
                    execution_time=execution_time,
                    confidence=result.confidence,
                    notes=result.notes,
                    plan_rationale=plan.assumptions,
                )
            except Exception:
                # If learning fails, continue without it
                pass

        # Add to conversation context
        conv_manager.add_turn(
            task=task,
            task_type=task_type.value,
            files=plan.files_to_read + (applied if 'applied' in locals() else [])
        )

        console.print(f"[cyan]Log saved to {log_path}[/cyan]")
        
        # Show performance stats if available
        try:
            from agent.performance import get_global_monitor
            monitor = get_global_monitor()
            stats = monitor.get_stats()
            if stats and not show_plan:
                console.print("\n[dim]Performance Stats:[/dim]")
                for operation, metrics in stats.items():
                    console.print(f"  {operation}: {metrics['avg']:.3f}s avg ({metrics['count']} calls)")
        except Exception:
            pass
        
        return

    if interactive_session:
        console.print(
            Panel(
                "Type another task and press Enter. Use Ctrl+C to exit.",
                title="Local Code Agent",
                border_style="green",
                expand=False,
            )
        )

    tasks: list[str] = []
    if initial_task.strip():
        tasks = [initial_task]
    else:
        console.print(
            Panel(
                f"[bold green]Welcome to Local Code Agent[/bold green]\n\n"
                f"Repository: [cyan]{repo_path}[/cyan]\n"
                f"Type your task below and press Enter.\n"
                f"Use Ctrl+C to exit.",
                title="Local Code Agent",
                border_style="green",
                expand=False,
            )
        )

    while True:
        if not tasks:
            try:
                next_task = console.input("→ ")
            except KeyboardInterrupt:
                raise typer.Exit()
            if not next_task.strip():
                if interactive_session:
                    continue
                raise typer.Exit()
            tasks = [next_task]

        current = tasks.pop(0)
        
        # Check for exit commands
        if current.lower() in ("exit", "quit", "q"):
            console.print("\n[dim]Goodbye![/dim]\n")
            raise typer.Exit(code=0)
        
        _run_one(current)
        if not interactive_session:
            return
        
        # Continue interactive loop
        console.print()  # Add spacing between tasks


@app.command(name="sessions")
def sessions_cmd(
    action: str = typer.Argument("list", help="Action: list, show, delete, export"),
    session_id: str = typer.Argument(None, help="Session ID (required for show, delete, export)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information"),
    full: bool = typer.Option(False, "--full", "-f", help="Show full content (for 'show' action)"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation (for 'delete' action)"),
    output: str = typer.Option("", "--output", "-o", help="Output path (for 'export' action)"),
) -> None:
    """
    Manage conversation sessions.
    
    Actions:
      list   - List all sessions
      show   - Show session details
      delete - Delete a session
      export - Export session to markdown
    """
    action = action.lower()
    
    if action == "list":
        list_sessions_cmd(detailed=detailed)
    elif action == "show":
        if not session_id:
            console.print("[red]Error: session_id is required for 'show' action[/red]")
            console.print("[dim]Usage: local-code-agent sessions show <session_id>[/dim]")
            raise typer.Exit(code=1)
        show_session_cmd(session_id, full=full)
    elif action == "delete":
        if not session_id:
            console.print("[red]Error: session_id is required for 'delete' action[/red]")
            console.print("[dim]Usage: local-code-agent sessions delete <session_id>[/dim]")
            raise typer.Exit(code=1)
        delete_session_cmd(session_id, force=force)
    elif action == "export":
        if not session_id:
            console.print("[red]Error: session_id is required for 'export' action[/red]")
            console.print("[dim]Usage: local-code-agent sessions export <session_id>[/dim]")
            raise typer.Exit(code=1)
        from pathlib import Path
        output_path = Path(output) if output else None
        export_session_cmd(session_id, output_path)
    else:
        console.print(f"[red]Error: Unknown action '{action}'[/red]")
        console.print("[dim]Valid actions: list, show, delete, export[/dim]")
        raise typer.Exit(code=1)


@app.command(name="find-usages")
def find_usages(
    symbol: str = typer.Argument(..., help="Symbol name to find usages for"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Specific file to search in"),
) -> None:
    """Find all usages of a symbol (function, class, etc.)."""
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        console.print("[red]Knowledge graph not available. Install dependencies.[/red]")
        raise typer.Exit(code=1)
    
    repo_root = Path(config.repo_root)
    try:
        graph = build_codebase_graph(repo_root)
        usages = graph.find_all_usages(symbol)
        
        if not usages:
            console.print(f"[yellow]No usages found for '{symbol}'[/yellow]")
            return
        
        console.print(f"[bold]Found {len(usages)} usage(s) of '{symbol}':[/bold]\n")
        for i, usage in enumerate(usages, 1):
            symbol_type = usage.symbol_type or "usage"
            console.print(f"{i}. {usage.file_path}:{usage.line} ({symbol_type})")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command(name="find-callers")
def find_callers(
    function: str = typer.Argument(..., help="Function name to find callers for"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Specific file containing the function"),
) -> None:
    """Find all functions that call a given function."""
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        console.print("[red]Knowledge graph not available. Install dependencies.[/red]")
        raise typer.Exit(code=1)
    
    repo_root = Path(config.repo_root)
    try:
        graph = build_codebase_graph(repo_root)
        call_graph_builder = build_call_graph(repo_root, graph)
        callers = call_graph_builder.find_callers(function, file)
        
        if not callers:
            console.print(f"[yellow]No callers found for '{function}'[/yellow]")
            return
        
        console.print(f"[bold]Found {len(callers)} caller(s) of '{function}':[/bold]\n")
        for i, caller in enumerate(callers, 1):
            caller_info = f"{caller.caller_function or 'module'}" if caller.caller_function else "module"
            console.print(f"{i}. {caller.caller_file}:{caller.caller_line} (in {caller_info})")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command(name="find-callees")
def find_callees(
    function: str = typer.Argument(..., help="Function name to find callees for"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Specific file containing the function"),
) -> None:
    """Find all functions called by a given function."""
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        console.print("[red]Knowledge graph not available. Install dependencies.[/red]")
        raise typer.Exit(code=1)
    
    repo_root = Path(config.repo_root)
    try:
        graph = build_codebase_graph(repo_root)
        call_graph_builder = build_call_graph(repo_root, graph)
        callees = call_graph_builder.find_callees(function, file)
        
        if not callees:
            console.print(f"[yellow]No callees found for '{function}'[/yellow]")
            return
        
        console.print(f"[bold]Found {len(callees)} function(s) called by '{function}':[/bold]\n")
        for i, callee in enumerate(callees, 1):
            console.print(f"{i}. {callee}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command(name="search-code")
def search_code(
    query: str = typer.Argument(..., help="Semantic search query"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of results to return"),
) -> None:
    """Semantically search the codebase."""
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        console.print("[red]Semantic search not available. Install dependencies.[/red]")
        raise typer.Exit(code=1)
    
    repo_root = Path(config.repo_root)
    try:
        search = create_semantic_search(repo_root)
        console.print(f"[dim]Indexing codebase...[/dim]")
        search.index_codebase()
        console.print(f"[dim]Searching for: '{query}'...[/dim]\n")
        
        results = search.search(query, top_k=top_k)
        
        if not results:
            console.print(f"[yellow]No results found for '{query}'[/yellow]")
            return
        
        console.print(f"[bold]Found {len(results)} result(s):[/bold]\n")
        for i, match in enumerate(results, 1):
            console.print(f"{i}. {match.file_path}:{match.start_line}-{match.end_line} (score: {match.similarity_score:.3f})")
            # Show snippet preview
            preview = match.content[:150].replace("\n", " ")
            console.print(f"   [dim]{preview}...[/dim]\n")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command(name="config")
def config_cmd(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    init: bool = typer.Option(False, "--init", help="Create a default config file"),
    validate: bool = typer.Option(False, "--validate", help="Validate the config file"),
) -> None:
    """Manage configuration file."""
    repo_root = str(config.repo_root)
    
    if init:
        try:
            config_path = create_default_config(repo_root)
            console.print(f"[green]✓[/green] Created config file at: {config_path}")
        except Exception as e:
            console.print(f"[red]Error creating config file: {e}[/red]")
            raise typer.Exit(code=1)
        return
    
    if validate:
        config_data = load_config(repo_root)
        is_valid, errors = validate_config(config_data)
        if is_valid:
            console.print("[green]✓[/green] Configuration is valid")
        else:
            console.print("[red]✗[/red] Configuration has errors:")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(code=1)
        return
    
    if show:
        config_path = find_config_file(repo_root)
        if config_path:
            console.print(f"[bold]Config file:[/bold] {config_path}")
            config_data = load_config(repo_root)
            from rich.json import JSON
            console.print(JSON.from_data(config_data))
        else:
            console.print("[yellow]No config file found.[/yellow]")
            console.print(f"Run [bold]local-code-agent config --init[/bold] to create one.")
        return
    
    # Default: show help
    console.print("[bold]Configuration Management[/bold]")
    console.print("\nCommands:")
    console.print("  [bold]config --show[/bold]      Show current configuration")
    console.print("  [bold]config --init[/bold]      Create default config file")
    console.print("  [bold]config --validate[/bold]  Validate config file")


if __name__ == "__main__":
    try:
        app()
    except LocalCodeAgentError as e:
        console.print(f"[red]Agent Error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Unexpected Error: {e}[/red]")
        raise typer.Exit(code=1)
