"""Task decomposition and multi-step execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

try:
    from core.llm import ask
    from agent.task_classifier import classify_task, TaskType
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.llm import ask
    from agent.task_classifier import classify_task, TaskType


class SubtaskStatus(Enum):
    """Status of a subtask."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """A single subtask in a decomposed task."""

    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)  # IDs of subtasks this depends on
    status: SubtaskStatus = SubtaskStatus.PENDING
    files_to_read: List[str] = field(default_factory=list)
    files_to_modify: List[str] = field(default_factory=list)
    result: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class DecomposedTask:
    """A task broken down into subtasks."""

    original_task: str
    subtasks: List[Subtask] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)  # Subtask IDs in execution order


def decompose_task(task: str, repo_files: List[str], use_llm: bool = True, llm_fn=None) -> DecomposedTask:
    """
    Break down a complex task into subtasks.

    Args:
        task: The original task description.
        repo_files: List of files in the repository.
        use_llm: Whether to use LLM for decomposition.
        llm_fn: Optional LLM function (for testing).

    Returns:
        DecomposedTask with subtasks and execution order.
    """
    # Simple heuristic-based decomposition for common patterns
    decomposed = DecomposedTask(original_task=task)

    # Check if task is simple enough (no decomposition needed)
    if _is_simple_task(task):
        subtask = Subtask(
            id="1",
            description=task,
            files_to_read=[],
            files_to_modify=[],
        )
        decomposed.subtasks = [subtask]
        decomposed.execution_order = ["1"]
        return decomposed

    # Try LLM-based decomposition if enabled
    if use_llm:
        try:
            llm_callable = llm_fn or ask
            decomposed = _decompose_with_llm(task, repo_files, llm_callable)
            if decomposed.subtasks:
                return decomposed
        except Exception:
            # Fall back to heuristic decomposition
            pass

    # Heuristic-based decomposition
    return _decompose_heuristic(task, repo_files)


def _is_simple_task(task: str) -> bool:
    """Check if a task is simple enough to not need decomposition."""
    task_lower = task.lower()
    
    # Simple tasks: single file operations, single changes
    simple_patterns = [
        "replace",
        "append",
        "insert",
        "delete",
        "fix",
        "add",
        "remove",
    ]
    
    # Count how many files/operations mentioned
    file_count = task.count(".py") + task.count(".md") + task.count(".txt")
    
    # If only one file and one operation, it's simple
    if file_count <= 1 and any(pattern in task_lower for pattern in simple_patterns):
        return True
    
    # Complex indicators
    complex_indicators = [
        "and",
        "then",
        "also",
        "multiple",
        "several",
        "all",
        "refactor",
        "restructure",
    ]
    
    if any(indicator in task_lower for indicator in complex_indicators):
        return False
    
    return True


def _decompose_with_llm(task: str, repo_files: List[str], llm_fn) -> DecomposedTask:
    """
    Use LLM to decompose task into subtasks.

    Args:
        task: Task description.
        repo_files: Available repository files.
        llm_fn: LLM function to call.

    Returns:
        DecomposedTask with LLM-generated subtasks.
    """
    file_list = "\n".join(repo_files[:50])  # Limit to first 50 files
    
    prompt = f"""Break down this coding task into smaller, sequential subtasks.

Task: {task}

Available files (sample):
{file_list}

Output format (JSON):
{{
  "subtasks": [
    {{
      "id": "1",
      "description": "First subtask description",
      "dependencies": [],
      "files_to_read": ["file1.py"],
      "files_to_modify": ["file1.py"]
    }},
    {{
      "id": "2",
      "description": "Second subtask description",
      "dependencies": ["1"],
      "files_to_read": ["file2.py"],
      "files_to_modify": ["file2.py"]
    }}
  ]
}}

Rules:
- Each subtask should be independent and testable
- Dependencies should be minimal (only when absolutely necessary)
- Files should be specific and relevant
- Keep subtasks focused (one change per subtask when possible)
- Order subtasks logically (dependencies first)

Return ONLY valid JSON, no other text."""

    try:
        response = llm_fn(prompt)
        # Try to extract JSON from response
        import json
        import re
        
        # Find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            
            decomposed = DecomposedTask(original_task=task)
            for st_data in data.get("subtasks", []):
                subtask = Subtask(
                    id=str(st_data.get("id", "")),
                    description=st_data.get("description", ""),
                    dependencies=[str(d) for d in st_data.get("dependencies", [])],
                    files_to_read=st_data.get("files_to_read", []),
                    files_to_modify=st_data.get("files_to_modify", []),
                )
                decomposed.subtasks.append(subtask)
            
            # Calculate execution order (topological sort)
            decomposed.execution_order = _topological_sort(decomposed.subtasks)
            
            return decomposed
    except Exception:
        pass
    
    # Return empty decomposition on failure
    return DecomposedTask(original_task=task)


def _decompose_heuristic(task: str, repo_files: List[str]) -> DecomposedTask:
    """
    Heuristic-based task decomposition.

    Args:
        task: Task description.
        repo_files: Available repository files.

    Returns:
        DecomposedTask with heuristically generated subtasks.
    """
    decomposed = DecomposedTask(original_task=task)
    task_lower = task.lower()
    
    # Pattern: "do X and Y" -> two subtasks
    if " and " in task_lower:
        parts = task_lower.split(" and ")
        for i, part in enumerate(parts, 1):
            subtask = Subtask(
                id=str(i),
                description=part.strip(),
                dependencies=[str(j) for j in range(1, i)],  # Each depends on previous
            )
            decomposed.subtasks.append(subtask)
    
    # Pattern: "do X then Y" -> two subtasks with dependency
    elif " then " in task_lower:
        parts = task_lower.split(" then ")
        for i, part in enumerate(parts, 1):
            subtask = Subtask(
                id=str(i),
                description=part.strip(),
                dependencies=[str(j) for j in range(1, i)],
            )
            decomposed.subtasks.append(subtask)
    
    # Pattern: Multiple files mentioned -> one subtask per file
    elif task.count(".py") > 1:
        # Extract file names
        import re
        files = re.findall(r'\S+\.py', task)
        for i, file in enumerate(files[:5], 1):  # Limit to 5 files
            subtask = Subtask(
                id=str(i),
                description=f"Update {file}",
                files_to_modify=[file],
            )
            decomposed.subtasks.append(subtask)
    
    else:
        # Single subtask
        subtask = Subtask(
            id="1",
            description=task,
        )
        decomposed.subtasks.append(subtask)
    
    # Set execution order
    decomposed.execution_order = [st.id for st in decomposed.subtasks]
    
    return decomposed


def _topological_sort(subtasks: List[Subtask]) -> List[str]:
    """
    Topologically sort subtasks by dependencies.

    Args:
        subtasks: List of subtasks with dependencies.

    Returns:
        List of subtask IDs in execution order.
    """
    # Build dependency graph
    in_degree: Dict[str, int] = {st.id: 0 for st in subtasks}
    dependencies: Dict[str, List[str]] = {st.id: [] for st in subtasks}
    
    for subtask in subtasks:
        for dep_id in subtask.dependencies:
            if dep_id in dependencies:
                dependencies[dep_id].append(subtask.id)
                in_degree[subtask.id] += 1
    
    # Kahn's algorithm
    queue = [st.id for st in subtasks if in_degree[st.id] == 0]
    result = []
    
    while queue:
        current = queue.pop(0)
        result.append(current)
        
        for dependent in dependencies.get(current, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    # Add any remaining subtasks (shouldn't happen if no cycles)
    remaining = [st.id for st in subtasks if st.id not in result]
    result.extend(remaining)
    
    return result


def get_ready_subtasks(decomposed: DecomposedTask) -> List[Subtask]:
    """
    Get subtasks that are ready to execute (dependencies satisfied).

    Args:
        decomposed: DecomposedTask to check.

    Returns:
        List of subtasks ready to execute.
    """
    ready = []
    for subtask in decomposed.subtasks:
        if subtask.status != SubtaskStatus.PENDING:
            continue
        
        # Check if all dependencies are completed
        deps_satisfied = all(
            any(st.id == dep_id and st.status == SubtaskStatus.COMPLETED
                for st in decomposed.subtasks)
            for dep_id in subtask.dependencies
        )
        
        if not subtask.dependencies or deps_satisfied:
            ready.append(subtask)
    
    return ready


if __name__ == "__main__":
    # Demo
    task = "add logging to main.py and update tests"
    repo_files = ["main.py", "test_main.py"]
    decomposed = decompose_task(task, repo_files, use_llm=False)
    print(f"Original: {decomposed.original_task}")
    print(f"Subtasks: {len(decomposed.subtasks)}")
    for st in decomposed.subtasks:
        print(f"  {st.id}: {st.description} (deps: {st.dependencies})")
    print(f"Execution order: {decomposed.execution_order}")
