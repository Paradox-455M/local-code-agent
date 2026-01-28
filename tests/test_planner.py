"""Unit tests for agent.planner module."""

from __future__ import annotations

import pytest
from pathlib import Path

try:
    from agent.planner import (
        build_plan,
        Plan,
        _score_file,
        _choose_files,
        _infer_path_from_task,
        _extract_filename,
        _prioritize_targets,
        _filter_by_dirs,
        to_json,
    )
except ImportError:
    pytest.skip("Missing dependencies", allow_module_level=True)


class TestScoreFile:
    """Tests for _score_file function."""

    def test_score_python_file(self):
        """Python files should get base score."""
        score, reasons = _score_file("test.py", ["test"])
        assert score > 0
        assert "python-file" in reasons

    def test_score_matches_keywords(self):
        """Files matching keywords should get higher scores."""
        score_match, _ = _score_file("test_runner.py", ["test", "runner"])
        score_no_match, _ = _score_file("other.py", ["test", "runner"])
        assert score_match > score_no_match

    def test_score_multiple_keywords(self):
        """Multiple keyword matches should increase score."""
        score_single, _ = _score_file("test.py", ["test"])
        score_double, _ = _score_file("test_runner.py", ["test", "runner"])
        assert score_double > score_single

    def test_score_depth_penalty(self):
        """Deeper paths should have lower scores."""
        score_shallow, _ = _score_file("test.py", ["test"])
        score_deep, _ = _score_file("deep/nested/path/test.py", ["test"])
        assert score_shallow > score_deep

    def test_score_no_keywords(self):
        """Files without keyword matches should have negative score."""
        score, reasons = _score_file("unrelated.py", ["test", "runner"])
        assert score < 0

    def test_score_markdown_file(self):
        """Markdown files should be scored."""
        score, reasons = _score_file("README.md", ["readme"])
        assert score > 0 or score < 0  # Should return a score either way


class TestChooseFiles:
    """Tests for _choose_files function."""

    def test_choose_files_with_keywords(self):
        """Should select files matching keywords."""
        repo_files = ["test_runner.py", "other.py", "test_helper.py", "unrelated.py"]
        chosen_read, chosen_modify, rationale = _choose_files("test runner", repo_files)
        assert len(chosen_read) > 0
        assert "test" in " ".join(chosen_read).lower() or "runner" in " ".join(chosen_read).lower()

    def test_choose_files_empty_task(self):
        """Empty task should return empty selection."""
        repo_files = ["test.py", "other.py"]
        chosen_read, chosen_modify, rationale = _choose_files("", repo_files)
        assert len(chosen_read) == 0
        assert len(chosen_modify) == 0

    def test_choose_files_no_matches(self):
        """No matches should return empty lists."""
        repo_files = ["unrelated.py", "other.py"]
        chosen_read, chosen_modify, rationale = _choose_files("completely different task", repo_files)
        assert len(chosen_read) == 0
        assert len(chosen_modify) == 0

    def test_choose_files_limits_results(self):
        """Should limit to top 10 files."""
        repo_files = [f"test_{i}.py" for i in range(20)]
        chosen_read, chosen_modify, rationale = _choose_files("test", repo_files)
        assert len(chosen_read) <= 10

    def test_choose_files_only_python_and_markdown(self):
        """Should only consider .py and .md files."""
        repo_files = ["test.py", "readme.md", "config.json", "data.txt"]
        chosen_read, chosen_modify, rationale = _choose_files("test", repo_files)
        # Should not include .json or .txt files
        assert all(f.endswith((".py", ".md")) for f in chosen_read)


class TestInferPathFromTask:
    """Tests for _infer_path_from_task function."""

    def test_infer_path_explicit_filename(self):
        """Should extract .py filename from task."""
        result = _infer_path_from_task("fix main.py")
        assert result is not None
        assert "main.py" in result

    def test_infer_path_no_filename(self):
        """Should return None if no .py file mentioned."""
        result = _infer_path_from_task("do something")
        assert result is None

    def test_infer_path_multiple_filenames(self):
        """Should extract first .py filename."""
        result = _infer_path_from_task("update test.py and main.py")
        assert result is not None
        assert ".py" in result

    def test_infer_path_absolute_path(self):
        """Should handle absolute paths."""
        result = _infer_path_from_task("fix /absolute/path/to/file.py")
        assert result is not None
        assert result.startswith("/")

    def test_infer_path_with_comma(self):
        """Should handle commas in task."""
        result = _infer_path_from_task("fix test.py, please")
        assert result is not None
        assert "test.py" in result


class TestExtractFilename:
    """Tests for _extract_filename function."""

    def test_extract_simple_filename(self):
        """Should extract .py filename."""
        result = _extract_filename("fix main.py")
        assert result == "main.py"

    def test_extract_with_punctuation(self):
        """Should handle punctuation."""
        result = _extract_filename("fix main.py, please!")
        assert result == "main.py"

    def test_extract_no_py_file(self):
        """Should return None if no .py file."""
        result = _extract_filename("do something")
        assert result is None

    def test_extract_multiple_files(self):
        """Should extract first .py file."""
        result = _extract_filename("update test.py and main.py")
        assert result == "test.py"


class TestPrioritizeTargets:
    """Tests for _prioritize_targets function."""

    def test_prioritize_with_targets(self):
        """Should prioritize files matching targets."""
        repo_files = ["test.py", "other.py", "test_helper.py"]
        result = _prioritize_targets(repo_files, ["test"])
        assert result[0].startswith("test")

    def test_prioritize_no_targets(self):
        """Should return original list if no targets."""
        repo_files = ["test.py", "other.py"]
        result = _prioritize_targets(repo_files, None)
        assert result == repo_files

    def test_prioritize_empty_targets(self):
        """Should return original list if empty targets."""
        repo_files = ["test.py", "other.py"]
        result = _prioritize_targets(repo_files, [])
        assert result == repo_files


class TestFilterByDirs:
    """Tests for _filter_by_dirs function."""

    def test_filter_single_dir(self):
        """Should filter files by directory."""
        repo_files = ["src/main.py", "src/test.py", "other.py"]
        result = _filter_by_dirs(repo_files, ["src"])
        assert all(f.startswith("src/") for f in result)
        assert len(result) == 2

    def test_filter_multiple_dirs(self):
        """Should filter by multiple directories."""
        repo_files = ["src/a.py", "tests/b.py", "other.py"]
        result = _filter_by_dirs(repo_files, ["src", "tests"])
        assert len(result) == 2
        assert all(any(f.startswith(d) for d in ["src/", "tests/"]) for f in result)

    def test_filter_no_matches(self):
        """Should return empty list if no matches."""
        repo_files = ["src/a.py", "tests/b.py"]
        result = _filter_by_dirs(repo_files, ["nonexistent"])
        assert len(result) == 0


class TestBuildPlan:
    """Tests for build_plan function."""

    def test_build_plan_basic(self):
        """Should create a valid plan."""
        repo_files = ["main.py", "test.py"]
        plan = build_plan("test task", repo_files)
        assert isinstance(plan, Plan)
        assert plan.goal == "test task"
        assert plan.risk_level in ["low", "medium", "high"]

    def test_build_plan_with_explicit_file(self):
        """Should include explicitly mentioned file."""
        repo_files = ["main.py", "test.py"]
        plan = build_plan("fix main.py", repo_files)
        # Planner may return absolute paths, so check if path ends with main.py
        assert any(f.endswith("main.py") for f in plan.files_to_read) or any(
            f.endswith("main.py") for f in plan.files_to_modify
        )

    def test_build_plan_modification_keywords(self):
        """Should mark files for modification with modification keywords."""
        repo_files = ["main.py"]
        plan = build_plan("fix main.py", repo_files)
        assert len(plan.files_to_modify) > 0

    def test_build_plan_read_only_keywords(self):
        """Should mark files as read-only without modification keywords."""
        repo_files = ["main.py"]
        plan = build_plan("explain main.py", repo_files)
        # Should have files_to_read but may not have files_to_modify
        assert len(plan.files_to_read) > 0

    def test_build_plan_has_steps(self):
        """Should include execution steps."""
        repo_files = ["main.py"]
        plan = build_plan("test task", repo_files)
        assert len(plan.steps) > 0
        assert all(isinstance(step, str) for step in plan.steps)

    def test_build_plan_has_assumptions(self):
        """Should include assumptions."""
        repo_files = ["main.py"]
        plan = build_plan("test task", repo_files)
        assert len(plan.assumptions) > 0

    def test_build_plan_with_targets(self):
        """Should prioritize target files."""
        repo_files = ["src/main.py", "other.py"]
        plan = build_plan("test task", repo_files, targets=["src"])
        # Files in src/ should be prioritized
        assert any("src" in f for f in plan.files_to_read) or len(plan.files_to_read) == 0

    def test_build_plan_with_dirs(self):
        """Should filter by directories."""
        repo_files = ["src/main.py", "tests/test.py", "other.py"]
        plan = build_plan("test task", repo_files, dirs=["src"])
        # Should only consider files in src/
        if plan.files_to_read:
            assert all("src" in f for f in plan.files_to_read)


class TestPlanValidation:
    """Tests for plan validation."""

    def test_plan_to_json(self):
        """Should serialize plan to JSON."""
        repo_files = ["main.py"]
        plan = build_plan("test task", repo_files)
        json_str = to_json(plan)
        assert isinstance(json_str, str)
        assert "goal" in json_str
        assert "test task" in json_str

    def test_plan_json_roundtrip(self):
        """Should be able to parse JSON back."""
        import json
        repo_files = ["main.py"]
        plan = build_plan("test task", repo_files)
        json_str = to_json(plan)
        parsed = json.loads(json_str)
        assert parsed["goal"] == "test task"
        assert isinstance(parsed["files_to_read"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
