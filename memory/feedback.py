"""Feedback collection and analysis."""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime

from .history import get_history, TaskRecord


def collect_feedback(
    task_id: Optional[int] = None,
    task: Optional[str] = None,
    feedback_text: Optional[str] = None,
    score: Optional[float] = None,
) -> None:
    """
    Collect user feedback on a task.

    Args:
        task_id: ID of the task record (if available).
        task: Task description (used to find record if task_id not provided).
        feedback_text: User feedback text.
        score: Feedback score (1.0 = perfect, 0.0 = poor).
    """
    history = get_history()

    if task_id:
        history.update_feedback(task_id, feedback_text or "", score or 0.5)
    elif task:
        # Find most recent matching task
        recent = history.get_recent_tasks(limit=20)
        for record in recent:
            if record.task == task:
                # Would need task_id, but for now just log
                # In practice, we'd store task_id when recording
                break


def analyze_feedback_patterns() -> Dict[str, Any]:
    """
    Analyze feedback patterns to identify improvement areas.

    Returns:
        Dictionary with analysis results.
    """
    history = get_history()
    recent = history.get_recent_tasks(limit=100)

    if not recent:
        return {
            "total_tasks": 0,
            "success_rate": 0.0,
            "avg_confidence": 0.0,
            "improvement_areas": [],
        }

    successful = [t for t in recent if t.success]
    with_feedback = [t for t in recent if t.feedback_score is not None]

    success_rate = len(successful) / len(recent) if recent else 0.0
    avg_confidence = sum(t.confidence for t in recent) / len(recent) if recent else 0.0
    avg_feedback = (
        sum(t.feedback_score for t in with_feedback) / len(with_feedback)
        if with_feedback
        else None
    )

    # Identify improvement areas
    improvement_areas = []
    if success_rate < 0.7:
        improvement_areas.append("Low success rate - review file selection strategy")
    if avg_confidence < 0.6:
        improvement_areas.append("Low confidence - improve context gathering")
    if avg_feedback and avg_feedback < 0.6:
        improvement_areas.append("Poor user feedback - review diff quality")

    return {
        "total_tasks": len(recent),
        "success_rate": success_rate,
        "avg_confidence": avg_confidence,
        "avg_feedback": avg_feedback,
        "improvement_areas": improvement_areas,
    }


if __name__ == "__main__":
    # Demo
    analysis = analyze_feedback_patterns()
    print(f"Success rate: {analysis['success_rate']:.2%}")
    print(f"Improvement areas: {analysis['improvement_areas']}")
