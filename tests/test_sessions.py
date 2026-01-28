"""Tests for session management functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from memory.conversation import ConversationManager, ConversationState, ConversationTurn


@pytest.fixture
def temp_sessions_dir():
    """Create a temporary sessions directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def conv_manager(temp_sessions_dir):
    """Create a conversation manager with temporary storage."""
    return ConversationManager(sessions_dir=temp_sessions_dir)


def test_start_session(conv_manager):
    """Test starting a new session."""
    session_id = conv_manager.start_session("test_session")
    
    assert session_id == "test_session"
    assert conv_manager.current_session_id == "test_session"
    assert "test_session" in conv_manager.sessions


def test_start_session_auto_name(conv_manager):
    """Test starting a session with auto-generated name."""
    session_id = conv_manager.start_session()
    
    assert session_id.startswith("session_")
    assert conv_manager.current_session_id == session_id


def test_add_turn(conv_manager):
    """Test adding a turn to a session."""
    conv_manager.start_session("test_session")
    conv_manager.add_turn(
        task="Fix bug in main.py",
        task_type="fix",
        response="Fixed the bug",
        files=["main.py"]
    )
    
    session = conv_manager.sessions["test_session"]
    assert len(session.turns) == 1
    assert session.turns[0].task == "Fix bug in main.py"
    assert session.turns[0].task_type == "fix"
    assert session.turns[0].response == "Fixed the bug"
    assert "main.py" in session.turns[0].files_referenced


def test_add_turn_without_session(conv_manager):
    """Test that add_turn creates a session if none exists."""
    conv_manager.add_turn(
        task="Add feature",
        task_type="feature",
        files=["feature.py"]
    )
    
    assert conv_manager.current_session_id is not None
    assert len(conv_manager.sessions[conv_manager.current_session_id].turns) == 1


def test_get_context(conv_manager):
    """Test getting context from a session."""
    conv_manager.start_session("test_session")
    conv_manager.add_turn("Task 1", "fix", files=["file1.py"])
    conv_manager.add_turn("Task 2", "feature", files=["file2.py"])
    
    context = conv_manager.get_context()
    
    assert context["has_context"] is True
    assert len(context["previous_tasks"]) == 2
    assert "file1.py" in context["context_files"]
    assert "file2.py" in context["context_files"]
    assert context["last_task_type"] == "feature"


def test_get_context_no_session(conv_manager):
    """Test getting context when no session exists."""
    context = conv_manager.get_context()
    
    assert context["has_context"] is False
    assert len(context["previous_tasks"]) == 0
    assert len(context["context_files"]) == 0


def test_is_follow_up(conv_manager):
    """Test detecting follow-up queries."""
    conv_manager.start_session("test_session")
    conv_manager.add_turn("Add login feature", "feature")
    
    assert conv_manager.is_follow_up("also add logout") is True
    assert conv_manager.is_follow_up("same for register page") is True
    assert conv_manager.is_follow_up("fix the other bug") is True
    assert conv_manager.is_follow_up("add a new feature") is False


def test_expand_follow_up(conv_manager):
    """Test expanding follow-up queries with context."""
    conv_manager.start_session("test_session")
    conv_manager.add_turn("Fix bug in auth.py", "fix", files=["auth.py"])
    
    expanded = conv_manager.expand_follow_up("also fix the same issue in user.py")
    
    assert "fix task on auth.py" in expanded or "auth.py" in expanded


def test_list_sessions(conv_manager, temp_sessions_dir):
    """Test listing all sessions."""
    conv_manager.start_session("session1")
    conv_manager.start_session("session2")
    
    sessions = conv_manager.list_sessions()
    
    assert len(sessions) >= 2
    assert "session1" in sessions
    assert "session2" in sessions


def test_delete_session(conv_manager, temp_sessions_dir):
    """Test deleting a session."""
    conv_manager.start_session("test_session")
    conv_manager.add_turn("Task", "fix")
    
    # Verify session exists
    assert "test_session" in conv_manager.list_sessions()
    
    # Delete session
    result = conv_manager.delete_session("test_session")
    
    assert result is True
    assert "test_session" not in conv_manager.list_sessions()


def test_session_persistence(temp_sessions_dir):
    """Test that sessions persist to disk."""
    # Create a session and add turns
    manager1 = ConversationManager(sessions_dir=temp_sessions_dir)
    manager1.start_session("persistent_session")
    manager1.add_turn("Task 1", "fix", files=["file1.py"])
    manager1.add_turn("Task 2", "feature", files=["file2.py"])
    
    # Create a new manager instance (simulating restart)
    manager2 = ConversationManager(sessions_dir=temp_sessions_dir)
    
    # Verify session was loaded
    assert "persistent_session" in manager2.list_sessions()
    
    # Load the session explicitly
    manager2.start_session("persistent_session")
    context = manager2.get_context()
    
    assert context["has_context"] is True
    assert len(context["previous_tasks"]) == 2
    assert "file1.py" in context["context_files"]
    assert "file2.py" in context["context_files"]


def test_session_file_format(conv_manager, temp_sessions_dir):
    """Test that session files are valid JSON."""
    conv_manager.start_session("test_session")
    conv_manager.add_turn("Task", "fix", response="Done", files=["test.py"])
    
    session_file = temp_sessions_dir / "test_session.json"
    assert session_file.exists()
    
    # Verify JSON is valid
    with session_file.open("r") as f:
        data = json.load(f)
    
    assert data["session_id"] == "test_session"
    assert len(data["turns"]) == 1
    assert data["turns"][0]["task"] == "Task"
    assert data["turns"][0]["task_type"] == "fix"


def test_update_preferences(conv_manager):
    """Test updating user preferences."""
    conv_manager.start_session("test_session")
    conv_manager.update_preferences("verbose", True)
    conv_manager.update_preferences("max_files", 10)
    
    session = conv_manager.sessions["test_session"]
    assert session.user_preferences["verbose"] is True
    assert session.user_preferences["max_files"] == 10


def test_context_accumulation(conv_manager):
    """Test that context accumulates across turns."""
    conv_manager.start_session("test_session")
    
    conv_manager.add_turn("Task 1", "fix", files=["file1.py"])
    assert len(conv_manager.get_context()["context_files"]) == 1
    
    conv_manager.add_turn("Task 2", "feature", files=["file2.py", "file3.py"])
    assert len(conv_manager.get_context()["context_files"]) == 3
    
    # Adding same file again shouldn't duplicate
    conv_manager.add_turn("Task 3", "refactor", files=["file1.py"])
    assert len(conv_manager.get_context()["context_files"]) == 3


def test_recent_tasks_limit(conv_manager):
    """Test that get_context returns only recent tasks."""
    conv_manager.start_session("test_session")
    
    # Add 10 tasks
    for i in range(10):
        conv_manager.add_turn(f"Task {i}", "fix")
    
    context = conv_manager.get_context()
    
    # Should only return last 5 tasks
    assert len(context["previous_tasks"]) == 5
    assert context["previous_tasks"][-1] == "Task 9"
    assert context["previous_tasks"][0] == "Task 5"


def test_empty_session_delete(conv_manager):
    """Test deleting a non-existent session."""
    result = conv_manager.delete_session("nonexistent")
    assert result is False


def test_multiple_sessions(conv_manager):
    """Test managing multiple sessions simultaneously."""
    # Create first session
    conv_manager.start_session("session1")
    conv_manager.add_turn("Task A", "fix", files=["a.py"])
    
    # Create second session
    conv_manager.start_session("session2")
    conv_manager.add_turn("Task B", "feature", files=["b.py"])
    
    # Verify both exist
    assert "session1" in conv_manager.sessions
    assert "session2" in conv_manager.sessions
    
    # Verify current session
    assert conv_manager.current_session_id == "session2"
    
    # Switch back to first session
    conv_manager.start_session("session1")
    assert conv_manager.current_session_id == "session1"
    
    # Verify context is correct
    context = conv_manager.get_context()
    assert "a.py" in context["context_files"]
    assert "b.py" not in context["context_files"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
