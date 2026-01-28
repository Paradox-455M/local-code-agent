"""Task history storage and pattern recognition."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from ..core.config import config
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.config import config


@dataclass
class TaskRecord:
    """Record of a completed task."""

    task: str
    task_type: str  # question, modify, create, etc.
    files_read: List[str]
    files_modified: List[str]
    success: bool
    timestamp: str
    execution_time: float
    confidence: float
    notes: List[str]
    plan_rationale: List[str] = None
    user_feedback: Optional[str] = None
    feedback_score: Optional[float] = None  # 1.0 = perfect, 0.0 = poor


class TaskHistory:
    """Manages task history storage and retrieval."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize task history database.

        Args:
            db_path: Path to SQLite database file. Defaults to repo_root/.lca/history.db
        """
        if db_path is None:
            repo_root = Path(config.repo_root)
            db_dir = repo_root / ".lca"
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "history.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    files_read TEXT,  -- JSON array
                    files_modified TEXT,  -- JSON array
                    success INTEGER NOT NULL,  -- 0 or 1
                    timestamp TEXT NOT NULL,
                    execution_time REAL,
                    confidence REAL,
                    notes TEXT,  -- JSON array
                    plan_rationale TEXT,  -- JSON array
                    user_feedback TEXT,
                    feedback_score REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_type ON tasks(task_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON tasks(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_success ON tasks(success)
            """)
            conn.commit()
        finally:
            conn.close()

    def record_task(self, record: TaskRecord) -> int:
        """
        Record a completed task.

        Args:
            record: TaskRecord to store.

        Returns:
            ID of the inserted record.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute("""
                INSERT INTO tasks (
                    task, task_type, files_read, files_modified, success,
                    timestamp, execution_time, confidence, notes, plan_rationale,
                    user_feedback, feedback_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.task,
                record.task_type,
                json.dumps(record.files_read),
                json.dumps(record.files_modified),
                1 if record.success else 0,
                record.timestamp,
                record.execution_time,
                record.confidence,
                json.dumps(record.notes),
                json.dumps(record.plan_rationale or []),
                record.user_feedback,
                record.feedback_score,
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_recent_tasks(self, limit: int = 10) -> List[TaskRecord]:
        """
        Get recent tasks.

        Args:
            limit: Maximum number of tasks to return.

        Returns:
            List of TaskRecord objects.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute("""
                SELECT * FROM tasks
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_record(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_similar_tasks(self, task: str, limit: int = 5) -> List[TaskRecord]:
        """
        Find tasks similar to the given task.

        Args:
            task: Task description to find similar tasks for.
            limit: Maximum number of similar tasks to return.

        Returns:
            List of similar TaskRecord objects.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Simple similarity: tasks containing similar keywords
            task_words = set(task.lower().split())
            cursor = conn.execute("SELECT * FROM tasks ORDER BY timestamp DESC LIMIT 100")
            candidates = [self._row_to_record(row) for row in cursor.fetchall()]

            scored = []
            for candidate in candidates:
                candidate_words = set(candidate.task.lower().split())
                # Jaccard similarity
                intersection = len(task_words & candidate_words)
                union = len(task_words | candidate_words)
                similarity = intersection / union if union > 0 else 0.0
                if similarity > 0.1:  # At least 10% similarity
                    scored.append((similarity, candidate))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [task for _, task in scored[:limit]]
        finally:
            conn.close()

    def get_successful_patterns(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze successful patterns.

        Args:
            task_type: Optional filter by task type.

        Returns:
            Dictionary with pattern statistics.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            if task_type:
                cursor = conn.execute("""
                    SELECT * FROM tasks WHERE success = 1 AND task_type = ?
                """, (task_type,))
            else:
                cursor = conn.execute("SELECT * FROM tasks WHERE success = 1")
            
            successful = [self._row_to_record(row) for row in cursor.fetchall()]
            
            if not successful:
                return {
                    "total": 0,
                    "common_files": [],
                    "avg_confidence": 0.0,
                    "common_patterns": [],
                }

            # Analyze common files
            file_counts: Dict[str, int] = {}
            total_confidence = 0.0
            for task in successful:
                for f in task.files_read + task.files_modified:
                    file_counts[f] = file_counts.get(f, 0) + 1
                total_confidence += task.confidence

            common_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                "total": len(successful),
                "common_files": [f for f, _ in common_files],
                "avg_confidence": total_confidence / len(successful),
                "common_patterns": self._extract_patterns(successful),
            }
        finally:
            conn.close()

    def _extract_patterns(self, tasks: List[TaskRecord]) -> List[str]:
        """Extract common patterns from successful tasks."""
        # Simple pattern extraction: common keywords
        keyword_counts: Dict[str, int] = {}
        for task in tasks:
            words = task.task.lower().split()
            for word in words:
                if len(word) > 3:  # Only meaningful words
                    keyword_counts[word] = keyword_counts.get(word, 0) + 1

        common_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, _ in common_keywords]

    def _row_to_record(self, row: tuple) -> TaskRecord:
        """Convert database row to TaskRecord."""
        return TaskRecord(
            task=row[1],
            task_type=row[2],
            files_read=json.loads(row[3] or "[]"),
            files_modified=json.loads(row[4] or "[]"),
            success=bool(row[5]),
            timestamp=row[6],
            execution_time=row[7] or 0.0,
            confidence=row[8] or 0.0,
            notes=json.loads(row[9] or "[]"),
            plan_rationale=json.loads(row[10] or "[]"),
            user_feedback=row[11],
            feedback_score=row[12],
        )

    def update_feedback(self, task_id: int, feedback: str, score: float) -> None:
        """
        Update user feedback for a task.

        Args:
            task_id: ID of the task record.
            feedback: User feedback text.
            score: Feedback score (1.0 = perfect, 0.0 = poor).
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                UPDATE tasks
                SET user_feedback = ?, feedback_score = ?
                WHERE id = ?
            """, (feedback, score, task_id))
            conn.commit()
        finally:
            conn.close()


# Global instance
_history: Optional[TaskHistory] = None


def get_history() -> TaskHistory:
    """Get or create the global task history instance."""
    global _history
    if _history is None:
        _history = TaskHistory()
    return _history


if __name__ == "__main__":
    # Demo
    history = TaskHistory()
    record = TaskRecord(
        task="fix bug in main.py",
        task_type="fix",
        files_read=["main.py"],
        files_modified=["main.py"],
        success=True,
        timestamp=datetime.now().isoformat(),
        execution_time=2.5,
        confidence=0.8,
        notes=["Fixed null pointer"],
    )
    task_id = history.record_task(record)
    print(f"Recorded task with ID: {task_id}")

    similar = history.get_similar_tasks("fix issue")
    print(f"Found {len(similar)} similar tasks")
