"""Pattern recognition and matching for task history."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from .history import TaskRecord, get_history


class PatternMatcher:
    """Matches current tasks to historical patterns."""

    def __init__(self):
        self.history = get_history()

    def find_similar_patterns(self, task: str, task_type: str) -> List[Tuple[TaskRecord, float]]:
        """
        Find similar historical patterns for a task.

        Args:
            task: Current task description.
            task_type: Type of task.

        Returns:
            List of (TaskRecord, similarity_score) tuples, sorted by similarity.
        """
        similar_tasks = self.history.get_similar_tasks(task, limit=10)
        
        # Filter by task type and success
        relevant = [
            t for t in similar_tasks
            if t.task_type == task_type and t.success
        ]

        # Score by similarity and success metrics
        scored: List[Tuple[TaskRecord, float]] = []
        task_words = set(task.lower().split())

        for record in relevant:
            # Jaccard similarity
            record_words = set(record.task.lower().split())
            intersection = len(task_words & record_words)
            union = len(task_words | record_words)
            similarity = intersection / union if union > 0 else 0.0

            # Boost by confidence and feedback score
            boost = record.confidence
            if record.feedback_score:
                boost += record.feedback_score * 0.5

            final_score = similarity * (1.0 + boost)
            scored.append((record, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:5]

    def suggest_files(self, task: str, task_type: str, repo_files: List[str]) -> List[str]:
        """
        Suggest files based on historical patterns.

        Args:
            task: Current task description.
            task_type: Type of task.
            repo_files: Available repository files.

        Returns:
            List of suggested file paths.
        """
        patterns = self.find_similar_patterns(task, task_type)
        
        if not patterns:
            return []

        # Collect files from successful similar tasks
        file_scores: Dict[str, float] = {}
        for record, similarity in patterns:
            for file_path in record.files_read + record.files_modified:
                if file_path in repo_files:
                    file_scores[file_path] = file_scores.get(file_path, 0.0) + similarity

        # Sort by score
        suggested = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return [file_path for file_path, _ in suggested[:10]]

    def get_strategy_recommendations(self, task_type: str) -> Dict[str, Any]:
        """
        Get strategy recommendations based on historical success.

        Args:
            task_type: Type of task.

        Returns:
            Dictionary with recommendations.
        """
        patterns = self.history.get_successful_patterns(task_type=task_type)
        
        recommendations = {
            "preferred_files": patterns.get("common_files", [])[:5],
            "avg_confidence_threshold": patterns.get("avg_confidence", 0.5),
            "common_keywords": patterns.get("common_patterns", []),
        }

        return recommendations


def learn_from_task(
    task: str,
    task_type: str,
    files_read: List[str],
    files_modified: List[str],
    success: bool,
    execution_time: float,
    confidence: float,
    notes: List[str],
    plan_rationale: List[str],
) -> None:
    """
    Learn from a completed task.

    Args:
        task: Task description.
        task_type: Type of task.
        files_read: Files that were read.
        files_modified: Files that were modified.
        success: Whether the task was successful.
        execution_time: Time taken to execute.
        confidence: Confidence score.
        notes: Execution notes.
        plan_rationale: Planning rationale.
    """
    from datetime import datetime
    from .history import TaskRecord

    history = get_history()
    record = TaskRecord(
        task=task,
        task_type=task_type,
        files_read=files_read,
        files_modified=files_modified,
        success=success,
        timestamp=datetime.now().isoformat(),
        execution_time=execution_time,
        confidence=confidence,
        notes=notes,
        plan_rationale=plan_rationale or [],
    )
    history.record_task(record)


if __name__ == "__main__":
    # Demo
    matcher = PatternMatcher()
    suggestions = matcher.suggest_files("fix bug", "fix", ["main.py", "test.py", "utils.py"])
    print(f"Suggested files: {suggestions}")
