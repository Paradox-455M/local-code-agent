"""Task classification and intent detection."""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Tuple


class TaskType(Enum):
    """Types of tasks the agent can handle."""

    QUESTION = "question"  # Answer a question (what, how, why, explain)
    MODIFY = "modify"  # Modify existing code
    CREATE = "create"  # Create new files
    DELETE = "delete"  # Delete files
    EXPLAIN = "explain"  # Explain code/file
    REFACTOR = "refactor"  # Refactor code
    FIX = "fix"  # Fix bugs/issues
    UNKNOWN = "unknown"  # Couldn't determine


def classify_task(task: str) -> Tuple[TaskType, float]:
    """
    Classify a task to determine its intent.

    Args:
        task: The user's task description.

    Returns:
        Tuple of (TaskType, confidence_score).
    """
    task_lower = task.lower().strip()

    # Question patterns (high confidence)
    question_patterns = [
        r"^what\s+(is|does|are|do|can|will)",
        r"^how\s+(does|do|can|will|to)",
        r"^why\s+(does|do|is|are)",
        r"^when\s+(does|do|is|are|will)",
        r"^where\s+(is|are|does|do)",
        r"^(explain|describe|tell me about|tell me)\s+",
        r"^(what|how|why)\s+.*\?$",  # Ends with question mark
    ]

    for pattern in question_patterns:
        if re.search(pattern, task_lower):
            return TaskType.QUESTION, 0.9

    # Explain patterns
    explain_patterns = [
        r"^explain\s+",
        r"^describe\s+",
        r"^what\s+(is|does)\s+.*\s+(do|mean|for)",
    ]
    for pattern in explain_patterns:
        if re.search(pattern, task_lower):
            return TaskType.EXPLAIN, 0.8

    # Modification patterns
    modify_keywords = ["fix", "change", "update", "modify", "edit", "alter", "adjust"]
    if any(task_lower.startswith(kw) or f" {kw} " in task_lower for kw in modify_keywords):
        return TaskType.MODIFY, 0.7

    # Creation patterns
    create_keywords = ["create", "add", "new", "make", "generate", "write"]
    if any(task_lower.startswith(kw) or f" {kw} " in task_lower for kw in create_keywords):
        # Check if it's about creating files
        if any(word in task_lower for word in ["file", "function", "class", "module", "script"]):
            return TaskType.CREATE, 0.8
        return TaskType.MODIFY, 0.6  # Could be modifying by adding

    # Deletion patterns
    delete_keywords = ["delete", "remove", "drop", "eliminate"]
    if any(task_lower.startswith(kw) or f" {kw} " in task_lower for kw in delete_keywords):
        return TaskType.DELETE, 0.8

    # Refactor patterns
    refactor_keywords = ["refactor", "restructure", "reorganize", "cleanup", "clean up"]
    if any(kw in task_lower for kw in refactor_keywords):
        return TaskType.REFACTOR, 0.7

    # Fix patterns
    fix_keywords = ["fix", "bug", "error", "issue", "problem", "broken"]
    if any(kw in task_lower for kw in fix_keywords):
        return TaskType.FIX, 0.7

    # Default: assume modification if unclear
    return TaskType.UNKNOWN, 0.3


def is_conversational_query(task: str) -> bool:
    """
    Check if a task is a conversational query that should be answered directly.

    Args:
        task: The user's task description.

    Returns:
        True if this is a question/explanation request.
    """
    task_type, confidence = classify_task(task)
    return task_type in (TaskType.QUESTION, TaskType.EXPLAIN) and confidence > 0.6


def should_modify_files(task: str) -> bool:
    """
    Check if a task requires file modifications.

    Args:
        task: The user's task description.

    Returns:
        True if files should be modified.
    """
    task_type, confidence = classify_task(task)
    return task_type in (TaskType.MODIFY, TaskType.CREATE, TaskType.DELETE, TaskType.REFACTOR, TaskType.FIX) and confidence > 0.5


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "what does this project do",
        "how does the planner work",
        "explain the architecture",
        "fix the bug in main.py",
        "add a new function",
        "create a test file",
        "delete unused code",
        "refactor the executor",
    ]

    for test in test_cases:
        task_type, confidence = classify_task(test)
        print(f"{test:30} -> {task_type.value:10} (confidence: {confidence:.2f})")
