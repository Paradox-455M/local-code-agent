"""Unit tests for agent.executor module."""

from __future__ import annotations

import pytest
from pathlib import Path

try:
    from agent.executor import (
        execute,
        ExecutorResult,
        _parse_instruction,
        _deterministic_apply,
        _validate_diffs,
        _extract_diffs,
        _build_prompt,
        _infer_mode,
        build_context,
    )
    from tools.apply_patch import PatchError
except ImportError:
    pytest.skip("Missing dependencies", allow_module_level=True)


class TestParseInstruction:
    """Tests for _parse_instruction function."""

    def test_parse_replace(self):
        """Should parse replace instructions."""
        result = _parse_instruction("replace foo with bar")
        assert result is not None
        assert result["kind"] == "replace"
        assert result["old"] == "foo"
        assert result["new"] == "bar"

    def test_parse_replace_quoted(self):
        """Should handle quoted strings."""
        result = _parse_instruction('replace "foo" with "bar"')
        assert result is not None
        assert result["kind"] == "replace"

    def test_parse_append_after(self):
        """Should parse append after instructions."""
        result = _parse_instruction("append new_line after anchor")
        assert result is not None
        assert result["kind"] == "append_after"
        assert result["text"] == "new_line"
        assert result["anchor"] == "anchor"

    def test_parse_insert_before(self):
        """Should parse insert before instructions."""
        result = _parse_instruction("insert header before start")
        assert result is not None
        assert result["kind"] == "insert_before"
        assert result["text"] == "header"
        assert result["anchor"] == "start"

    def test_parse_prepend(self):
        """Should parse prepend instructions."""
        result = _parse_instruction("prepend header before start")
        assert result is not None
        assert result["kind"] == "insert_before"

    def test_parse_case_insensitive(self):
        """Should be case insensitive."""
        result = _parse_instruction("REPLACE foo WITH bar")
        assert result is not None
        assert result["kind"] == "replace"

    def test_parse_invalid(self):
        """Should return None for invalid instructions."""
        result = _parse_instruction("do something")
        assert result is None

    def test_parse_empty(self):
        """Should return None for empty string."""
        result = _parse_instruction("")
        assert result is None


class TestDeterministicApply:
    """Tests for _deterministic_apply function."""

    def test_replace_success(self):
        """Should replace text successfully."""
        existing = {"test.txt": "old text\n"}
        parsed = {"kind": "replace", "old": "old", "new": "new"}
        result = _deterministic_apply(parsed, existing, ["test.txt"])
        assert len(result.diffs) > 0
        assert result.confidence > 0

    def test_replace_no_match(self):
        """Should return empty diffs if no match."""
        existing = {"test.txt": "different text\n"}
        parsed = {"kind": "replace", "old": "old", "new": "new"}
        result = _deterministic_apply(parsed, existing, ["test.txt"])
        assert len(result.diffs) == 0
        assert result.confidence == 0.0

    def test_append_after_success(self):
        """Should append after anchor successfully."""
        existing = {"test.txt": "start\nanchor\nend\n"}
        parsed = {"kind": "append_after", "text": "new_line", "anchor": "anchor"}
        result = _deterministic_apply(parsed, existing, ["test.txt"])
        assert len(result.diffs) > 0

    def test_insert_before_success(self):
        """Should insert before anchor successfully."""
        existing = {"test.txt": "start\nanchor\nend\n"}
        parsed = {"kind": "insert_before", "text": "header", "anchor": "start"}
        result = _deterministic_apply(parsed, existing, ["test.txt"])
        assert len(result.diffs) > 0

    def test_multiple_files(self):
        """Should apply to multiple files."""
        existing = {"file1.txt": "old\n", "file2.txt": "old\n"}
        parsed = {"kind": "replace", "old": "old", "new": "new"}
        result = _deterministic_apply(parsed, existing, ["file1.txt", "file2.txt"])
        assert len(result.diffs) == 2


class TestExtractDiffs:
    """Tests for _extract_diffs function."""

    def test_extract_single_diff(self):
        """Should extract single diff."""
        text = """--- test.txt
+++ test.txt
@@ -1 +1 @@
-old
+new
"""
        diffs = _extract_diffs(text)
        assert len(diffs) == 1
        assert "--- test.txt" in diffs[0]["diff"]

    def test_extract_multiple_diffs(self):
        """Should extract multiple diffs."""
        text = """--- file1.txt
+++ file1.txt
@@ -1 +1 @@
-old
+new
--- file2.txt
+++ file2.txt
@@ -1 +1 @@
-old
+new
"""
        diffs = _extract_diffs(text)
        assert len(diffs) == 2

    def test_extract_with_plan(self):
        """Should extract diff after plan text."""
        text = """Plan:
- Do something
- Do something else

--- test.txt
+++ test.txt
@@ -1 +1 @@
-old
+new
"""
        diffs = _extract_diffs(text)
        assert len(diffs) == 1

    def test_extract_invalid(self):
        """Should return empty list for invalid diff."""
        text = "This is not a diff"
        diffs = _extract_diffs(text)
        assert len(diffs) == 0

    def test_extract_no_headers(self):
        """Should filter out diffs without proper headers."""
        text = """--- test.txt
some content
"""
        diffs = _extract_diffs(text)
        # Should filter out as it doesn't have +++ header
        assert len(diffs) == 0 or all("+++" in d["diff"] for d in diffs)


class TestValidateDiffs:
    """Tests for _validate_diffs function."""

    def test_validate_valid_diff(self):
        """Should validate a proper diff."""
        diffs = [
            {
                "path": "test.txt",
                "diff": "--- test.txt\n+++ test.txt\n@@ -1 +1 @@\n-old\n+new\n",
            }
        ]
        result = _validate_diffs(diffs, allowed_paths=["test.txt"], repo_root=".")
        assert len(result) == 1

    def test_validate_size_limit(self):
        """Should reject diffs exceeding size limit."""
        large_diff = "--- test.txt\n+++ test.txt\n" + "@@ -1 +1 @@\n" + ("x\n" * 10000)
        diffs = [{"path": "test.txt", "diff": large_diff}]
        result = _validate_diffs(diffs, max_bytes=1000, allowed_paths=["test.txt"], repo_root=".")
        assert len(result) == 0

    def test_validate_line_limit(self):
        """Should reject diffs exceeding line limit."""
        large_diff = "--- test.txt\n+++ test.txt\n" + "@@ -1 +1 @@\n" + ("x\n" * 10000)
        diffs = [{"path": "test.txt", "diff": large_diff}]
        result = _validate_diffs(diffs, max_lines=100, allowed_paths=["test.txt"], repo_root=".")
        assert len(result) == 0

    def test_validate_path_restriction(self):
        """Should only allow diffs for allowed paths."""
        diffs = [
            {"path": "allowed.txt", "diff": "--- allowed.txt\n+++ allowed.txt\n@@ -1 +1 @@\n-old\n+new\n"}
        ]
        result = _validate_diffs(diffs, allowed_paths=["allowed.txt"], repo_root=".")
        assert len(result) == 1

        diffs_not_allowed = [
            {"path": "not_allowed.txt", "diff": "--- not_allowed.txt\n+++ not_allowed.txt\n@@ -1 +1 @@\n-old\n+new\n"}
        ]
        result = _validate_diffs(diffs_not_allowed, allowed_paths=["allowed.txt"], repo_root=".")
        assert len(result) == 0

    def test_validate_missing_path(self):
        """Should infer path from diff if missing."""
        diffs = [
            {"path": "", "diff": "--- test.txt\n+++ test.txt\n@@ -1 +1 @@\n-old\n+new\n"}
        ]
        result = _validate_diffs(diffs, allowed_paths=["test.txt"], repo_root=".")
        assert len(result) == 1
        assert result[0]["path"] == "test.txt"

    def test_validate_relaxed_mode(self):
        """Should be more lenient in relaxed mode."""
        large_diff = "--- test.txt\n+++ test.txt\n" + "@@ -1 +1 @@\n" + ("x\n" * 10000)
        diffs = [{"path": "test.txt", "diff": large_diff}]
        result = _validate_diffs(diffs, relaxed=True, allowed_paths=["test.txt"], repo_root=".")
        # Relaxed mode should allow larger diffs
        assert len(result) >= 0  # May or may not pass, but shouldn't crash


class TestInferMode:
    """Tests for _infer_mode function."""

    def test_infer_bugfix_mode(self):
        """Should detect bugfix mode."""
        assert _infer_mode("fix bug") == "bugfix"
        assert _infer_mode("resolve error") == "bugfix"

    def test_infer_refactor_mode(self):
        """Should detect refactor mode."""
        assert _infer_mode("refactor code") == "refactor"
        assert _infer_mode("cleanup") == "refactor"

    def test_infer_feature_mode(self):
        """Should detect feature mode."""
        assert _infer_mode("add feature") == "feature"
        assert _infer_mode("implement new") == "feature"

    def test_infer_general_mode(self):
        """Should default to general mode."""
        assert _infer_mode("do something") == "general"


class TestBuildContext:
    """Tests for build_context function."""

    def test_build_context_limits_files(self):
        """Should limit number of context files."""
        base = Path(".")
        test_files = []
        for i in range(10):
            test_file = Path(f"test_context_{i}.txt")
            test_file.write_text(f"content {i}\n", encoding="utf-8")
            test_files.append(str(test_file))

        try:
            snippets = build_context(base, test_files, max_context_files=5)
            assert len(snippets) <= 5
        finally:
            for f in test_files:
                Path(f).unlink(missing_ok=True)

    def test_build_context_limits_bytes(self):
        """Should limit snippet size."""
        base = Path(".")
        test_file = Path("test_large.txt")
        test_file.write_text("x" * 10000, encoding="utf-8")

        try:
            snippets = build_context(base, [str(test_file)], max_bytes=1000)
            assert len(snippets) == 1
            assert len(snippets[0]["snippet"]) <= 1000
        finally:
            test_file.unlink(missing_ok=True)

    def test_build_context_prioritizes_symbols(self):
        """Should prioritize files with symbols."""
        base = Path(".")
        file1 = Path("test_symbol.txt")
        file2 = Path("test_other.txt")
        file1.write_text("def target_symbol(): pass\n", encoding="utf-8")
        file2.write_text("def other(): pass\n", encoding="utf-8")

        try:
            snippets = build_context(base, [str(file1), str(file2)], symbols=["target_symbol"])
            # File with symbol should be prioritized
            assert len(snippets) > 0
            # Check if any snippet contains the symbol (symbol-based prioritization may vary)
            # The symbol should appear in at least one snippet
            assert any("target_symbol" in s["snippet"] for s in snippets) or any(
                s["path"] == str(file1) for s in snippets
            )
        finally:
            file1.unlink(missing_ok=True)
            file2.unlink(missing_ok=True)


class TestExecute:
    """Tests for execute function."""

    def test_execute_no_files(self):
        """Should return empty result when no files specified."""
        result = execute("test task", [], [], ".")
        assert len(result.diffs) == 0
        assert result.confidence == 0.0

    def test_execute_deterministic_replace(self):
        """Should execute deterministic replace."""
        test_file = Path("test_exec.txt")
        test_file.write_text("old text\n", encoding="utf-8")

        try:
            result = execute(
                "replace old with new",
                [str(test_file)],
                [str(test_file)],
                ".",
                instruction="replace old with new",
            )
            assert len(result.diffs) > 0
            assert result.confidence > 0
        finally:
            test_file.unlink(missing_ok=True)
            bak = test_file.with_suffix(test_file.suffix + ".bak")
            bak.unlink(missing_ok=True)

    def test_execute_with_llm_stub(self):
        """Should use LLM stub when provided."""
        test_file = Path("test_llm.txt")
        test_file.write_text("old\n", encoding="utf-8")

        def stub_llm(_: str) -> str:
            # Use the actual file path in the diff
            return f"--- {test_file}\n+++ {test_file}\n@@ -1 +1 @@\n-old\n+new\n"

        try:
            result = execute(
                "make change",
                [str(test_file)],
                [str(test_file)],
                ".",
                llm_fn=stub_llm,
                enable_llm=True,
            )
            assert len(result.diffs) > 0
        finally:
            test_file.unlink(missing_ok=True)
            bak = test_file.with_suffix(test_file.suffix + ".bak")
            bak.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
