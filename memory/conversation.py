"""Conversation state management and context."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from datetime import datetime


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    task: str
    task_type: str
    response: Optional[str] = None
    files_referenced: List[str] = field(default_factory=list)
    symbols_referenced: List[str] = field(default_factory=list)  # Functions, classes mentioned
    code_entities: Dict[str, Any] = field(default_factory=dict)  # Track code entities
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationState:
    """State of an ongoing conversation."""

    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    context_files: Set[str] = field(default_factory=set)
    context_symbols: Set[str] = field(default_factory=set)  # Symbols mentioned
    code_entities: Dict[str, Any] = field(default_factory=dict)  # Track entities: functions, classes, etc.
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    last_task_type: Optional[str] = None
    last_files: List[str] = field(default_factory=list)
    last_symbols: List[str] = field(default_factory=list)


class ConversationManager:
    """Manages conversation state across multiple turns with persistence."""

    def __init__(self, sessions_dir: Optional[Path] = None):
        self.sessions: Dict[str, ConversationState] = {}
        self.current_session_id: Optional[str] = None
        self.sessions_dir = sessions_dir or Path(".lca/sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._load_all_sessions()

    def _save_session(self, session_id: str) -> None:
        """Save a session to disk."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session_file = self.sessions_dir / f"{session_id}.json"
        
        # Convert to dict for JSON serialization
        session_dict = {
            "session_id": session.session_id,
            "turns": [
                {
                    "task": turn.task,
                    "task_type": turn.task_type,
                    "response": turn.response,
                    "files_referenced": turn.files_referenced,
                    "symbols_referenced": getattr(turn, "symbols_referenced", []),
                    "code_entities": getattr(turn, "code_entities", {}),
                    "timestamp": turn.timestamp,
                }
                for turn in session.turns
            ],
            "context_files": list(session.context_files),
            "context_symbols": list(getattr(session, "context_symbols", set())),
            "code_entities": getattr(session, "code_entities", {}),
            "user_preferences": session.user_preferences,
            "last_task_type": session.last_task_type,
            "last_files": session.last_files,
            "last_symbols": getattr(session, "last_symbols", []),
        }
        
        try:
            with session_file.open("w", encoding="utf-8") as f:
                json.dump(session_dict, f, indent=2)
        except Exception as e:
            # Silently fail - persistence is optional
            pass

    def _load_session(self, session_id: str) -> bool:
        """Load a session from disk. Returns True if loaded successfully."""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return False
        
        try:
            with session_file.open("r", encoding="utf-8") as f:
                session_dict = json.load(f)
            
            # Reconstruct session
            session = ConversationState(
                session_id=session_dict["session_id"],
                context_files=set(session_dict.get("context_files", [])),
                context_symbols=set(session_dict.get("context_symbols", [])),
                code_entities=session_dict.get("code_entities", {}),
                user_preferences=session_dict.get("user_preferences", {}),
                last_task_type=session_dict.get("last_task_type"),
                last_files=session_dict.get("last_files", []),
                last_symbols=session_dict.get("last_symbols", []),
            )
            
            # Reconstruct turns
            for turn_dict in session_dict.get("turns", []):
                turn = ConversationTurn(
                    task=turn_dict["task"],
                    task_type=turn_dict["task_type"],
                    response=turn_dict.get("response"),
                    files_referenced=turn_dict.get("files_referenced", []),
                    symbols_referenced=turn_dict.get("symbols_referenced", []),
                    code_entities=turn_dict.get("code_entities", {}),
                    timestamp=turn_dict.get("timestamp", datetime.now().isoformat()),
                )
                session.turns.append(turn)
            
            self.sessions[session_id] = session
            return True
        except Exception:
            return False

    def _load_all_sessions(self) -> None:
        """Load all existing sessions from disk."""
        if not self.sessions_dir.exists():
            return
        
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem
            if session_id not in self.sessions:
                self._load_session(session_id)

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.

        Args:
            session_id: Optional session ID. If not provided, generates one.

        Returns:
            Session ID.
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if session_id not in self.sessions:
            # Try to load existing session
            if not self._load_session(session_id):
                self.sessions[session_id] = ConversationState(session_id=session_id)
        
        self.current_session_id = session_id
        self._save_session(session_id)
        return session_id

    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        sessions = []
        if self.sessions_dir.exists():
            for session_file in self.sessions_dir.glob("*.json"):
                sessions.append(session_file.stem)
        return sorted(sessions)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted successfully."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            try:
                session_file.unlink()
                return True
            except Exception:
                return False
        return False

    def add_turn(
        self,
        task: str,
        task_type: str,
        response: Optional[str] = None,
        files: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        code_entities: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a turn to the current conversation.

        Args:
            task: User's task/query.
            task_type: Type of task.
            response: Agent's response (if available).
            files: Files referenced in this turn.
            symbols: Symbols (functions, classes) referenced.
            code_entities: Code entities (functions, classes) with metadata.
        """
        if not self.current_session_id:
            self.start_session()

        session = self.sessions[self.current_session_id]
        turn = ConversationTurn(
            task=task,
            task_type=task_type,
            response=response,
            files_referenced=files or [],
            symbols_referenced=symbols or [],
            code_entities=code_entities or {},
        )
        session.turns.append(turn)
        session.last_task_type = task_type
        session.last_files = files or []
        session.last_symbols = symbols or []
        if files:
            session.context_files.update(files)
        if symbols:
            session.context_symbols.update(symbols)
        if code_entities:
            session.code_entities.update(code_entities)
        
        # Auto-save after adding turn
        self._save_session(self.current_session_id)

    def get_context(self) -> Dict[str, Any]:
        """
        Get conversation context for the current session.

        Returns:
            Dictionary with context information.
        """
        if not self.current_session_id:
            return {
                "has_context": False,
                "previous_tasks": [],
                "context_files": [],
                "context_symbols": [],
                "code_entities": {},
            }

        session = self.sessions[self.current_session_id]
        return {
            "has_context": len(session.turns) > 0,
            "previous_tasks": [t.task for t in session.turns[-5:]],  # Last 5 tasks
            "context_files": list(session.context_files),
            "context_symbols": list(session.context_symbols),
            "code_entities": session.code_entities,
            "last_task_type": session.last_task_type,
            "last_files": session.last_files,
            "last_symbols": session.last_symbols,
            "user_preferences": session.user_preferences,
        }
    
    def resolve_reference(self, reference: str) -> Optional[Any]:
        """
        Resolve a reference like "the same function", "that file", etc.
        
        Args:
            reference: Reference string to resolve.
        
        Returns:
            Resolved entity or None.
        """
        if not self.current_session_id:
            return None
        
        session = self.sessions[self.current_session_id]
        if not session.turns:
            return None
        
        last_turn = session.turns[-1]
        reference_lower = reference.lower()
        
        # Resolve common references
        if "same function" in reference_lower or "that function" in reference_lower:
            if last_turn.symbols_referenced:
                return last_turn.symbols_referenced[-1]
        
        if "same file" in reference_lower or "that file" in reference_lower:
            if last_turn.files_referenced:
                return last_turn.files_referenced[-1]
        
        if "previous" in reference_lower or "last" in reference_lower:
            if "file" in reference_lower and last_turn.files_referenced:
                return last_turn.files_referenced[-1]
            if "function" in reference_lower and last_turn.symbols_referenced:
                return last_turn.symbols_referenced[-1]
        
        return None
    
    def expand_task_with_context(self, task: str) -> str:
        """
        Expand task with conversation context and resolve references.
        
        Args:
            task: Original task.
        
        Returns:
            Expanded task with context.
        """
        if not self.current_session_id:
            return task
        
        session = self.sessions[self.current_session_id]
        if not session.turns:
            return task
        
        expanded = task
        
        # Resolve references
        references = [
            ("the same function", self.resolve_reference("the same function")),
            ("that function", self.resolve_reference("that function")),
            ("the same file", self.resolve_reference("the same file")),
            ("that file", self.resolve_reference("that file")),
        ]
        
        for ref_text, resolved in references:
            if resolved and ref_text in task.lower():
                expanded = expanded.replace(ref_text, str(resolved))
                expanded = expanded.replace(ref_text.title(), str(resolved))
        
        # Add context from last turn if it's a follow-up
        if self.is_follow_up(task):
            context = self.get_context()
            if context["last_files"]:
                expanded += f" (context: working on {', '.join(context['last_files'][:2])})"
            if context["last_symbols"]:
                expanded += f" (symbols: {', '.join(context['last_symbols'][:2])})"
        
        return expanded

    def is_follow_up(self, task: str) -> bool:
        """
        Check if a task is a follow-up to previous conversation.

        Args:
            task: Current task.

        Returns:
            True if this appears to be a follow-up.
        """
        if not self.current_session_id:
            return False

        session = self.sessions[self.current_session_id]
        if not session.turns:
            return False

        # Check for follow-up indicators
        follow_up_indicators = [
            "same",
            "also",
            "too",
            "other",
            "another",
            "again",
            "continue",
            "next",
        ]
        task_lower = task.lower()
        return any(indicator in task_lower for indicator in follow_up_indicators)

    def expand_follow_up(self, task: str) -> str:
        """
        Expand a follow-up task with context from previous turns.

        Args:
            task: Current task (may be a follow-up).

        Returns:
            Expanded task with context.
        """
        if not self.is_follow_up(task):
            return task

        context = self.get_context()
        if not context["has_context"]:
            return task

        # Add context from last task
        if context["last_task_type"] and context["last_files"]:
            expanded = f"{task} (context: {context['last_task_type']} task on {', '.join(context['last_files'][:3])})"
            return expanded

        return task

    def update_preferences(self, key: str, value: Any) -> None:
        """
        Update user preferences.

        Args:
            key: Preference key.
            value: Preference value.
        """
        if not self.current_session_id:
            self.start_session()

        self.sessions[self.current_session_id].user_preferences[key] = value
        self._save_session(self.current_session_id)


# Global conversation manager
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get or create the global conversation manager."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


if __name__ == "__main__":
    # Demo
    manager = get_conversation_manager()
    manager.start_session()
    manager.add_turn("fix bug in main.py", "fix", files=["main.py"])
    context = manager.get_context()
    print(f"Has context: {context['has_context']}")
    print(f"Last files: {context['last_files']}")

    is_follow = manager.is_follow_up("fix the same issue in other files")
    print(f"Is follow-up: {is_follow}")
