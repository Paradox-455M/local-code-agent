"""Unit tests for tools.apply_patch module."""

from __future__ import annotations

import pytest
from pathlib import Path

try:
    from tools.apply_patch import (
        apply_patch,
        PatchError,
        _parse_diff,
        _apply_hunks,
        _is_binary,
        FilePatch,
    )
except ImportError:
    pytest.skip("Missing dependencies", allow_module_level=True)


class TestApplyHunks:
    """Tests for _apply_hunks function."""

    def test_apply_simple_addition(self):
        """Should add lines correctly."""
        orig = ["line1", "line2"]  # No newlines - matches splitlines(keepends=False)
        hunks = [["@@ -1,2 +1,3 @@", " line1", "+new_line", " line2"]]
        result = _apply_hunks(orig, hunks)
        assert len(result) == 3
        assert "new_line" in result[1]

    def test_apply_simple_deletion(self):
        """Should delete lines correctly."""
        orig = ["line1", "line2", "line3"]  # No newlines
        hunks = [["@@ -1,3 +1,2 @@", " line1", "-line2", " line3"]]
        result = _apply_hunks(orig, hunks)
        assert len(result) == 2
        assert "line2" not in "".join(result)

    def test_apply_replacement(self):
        """Should replace lines correctly."""
        orig = ["old"]  # No newline
        hunks = [["@@ -1 +1 @@", "-old", "+new"]]
        result = _apply_hunks(orig, hunks)
        assert result[0] == "new"

    def test_apply_multiple_hunks(self):
        """Should apply multiple hunks."""
        orig = ["line1", "line2", "line3"]  # No newlines
        hunks = [
            ["@@ -1,1 +1,2 @@", " line1", "+insert1"],
            ["@@ -3,1 +4,2 @@", " line3", "+insert2"],
        ]
        result = _apply_hunks(orig, hunks)
        assert "insert1" in "".join(result)
        assert "insert2" in "".join(result)

    def test_apply_context_mismatch(self):
        """Should raise error on context mismatch."""
        orig = ["different\n"]
        hunks = [["@@ -1 +1 @@", " expected", "+new"]]
        with pytest.raises(PatchError, match="Context mismatch"):
            _apply_hunks(orig, hunks)

    def test_apply_invalid_hunk_header(self):
        """Should raise error on invalid hunk header."""
        orig = ["line1\n"]
        hunks = [["@@ invalid @@", " line1"]]
        with pytest.raises(PatchError, match="Invalid hunk header"):
            _apply_hunks(orig, hunks)


class TestParseDiff:
    """Tests for _parse_diff function."""

    def test_parse_single_file_diff(self):
        """Should parse single file diff."""
        diff = """--- test.txt
+++ test.txt
@@ -1 +1 @@
-old
+new
"""
        patches = _parse_diff(diff)
        assert len(patches) == 1
        assert patches[0].path.name == "test.txt"

    def test_parse_multi_file_diff(self):
        """Should parse multi-file diff."""
        diff = """--- file1.txt
+++ file1.txt
@@ -1 +1 @@
-old1
+new1
--- file2.txt
+++ file2.txt
@@ -1 +1 @@
-old2
+new2
"""
        patches = _parse_diff(diff)
        assert len(patches) == 2

    def test_parse_new_file(self):
        """Should parse diff for new file."""
        diff = """--- /dev/null
+++ new.txt
@@ -0,0 +1 @@
+content
"""
        patches = _parse_diff(diff)
        assert len(patches) == 1
        assert "new.txt" in str(patches[0].path)

    def test_parse_invalid_diff(self):
        """Should raise error on invalid diff."""
        diff = "not a valid diff"
        with pytest.raises(PatchError):
            _parse_diff(diff)

    def test_parse_no_hunks(self):
        """Should raise error if no hunks found."""
        diff = """--- test.txt
+++ test.txt
"""
        with pytest.raises(PatchError, match="No hunks found"):
            _parse_diff(diff)


class TestApplyPatch:
    """Tests for apply_patch function."""

    def test_apply_simple_patch(self):
        """Should apply simple patch successfully."""
        test_file = Path("test_apply.txt")
        test_file.write_text("old\n", encoding="utf-8")

        diff = f"""--- {test_file}
+++ {test_file}
@@ -1 +1 @@
-old
+new
"""
        try:
            modified = apply_patch(diff)
            assert str(test_file.resolve()) in modified
            assert test_file.read_text() == "new\n"
        finally:
            test_file.unlink(missing_ok=True)
            bak = test_file.with_suffix(test_file.suffix + ".bak")
            bak.unlink(missing_ok=True)

    def test_apply_creates_backup(self):
        """Should create backup file."""
        test_file = Path("test_backup.txt")
        test_file.write_text("original\n", encoding="utf-8")
        bak_file = test_file.with_suffix(test_file.suffix + ".bak")

        diff = f"""--- {test_file}
+++ {test_file}
@@ -1 +1 @@
-original
+modified
"""
        try:
            apply_patch(diff, keep_backup=True)
            assert bak_file.exists()
            assert bak_file.read_text() == "original\n"
        finally:
            test_file.unlink(missing_ok=True)
            bak_file.unlink(missing_ok=True)

    def test_apply_new_file(self):
        """Should create new file from diff."""
        test_file = Path("test_new.txt")

        diff = f"""--- /dev/null
+++ {test_file}
@@ -0,0 +1 @@
+new content
"""
        try:
            modified = apply_patch(diff)
            assert test_file.exists()
            assert "new content" in test_file.read_text()
        finally:
            test_file.unlink(missing_ok=True)

    def test_apply_rollback_on_failure(self):
        """Should rollback all changes on failure."""
        test_file1 = Path("test_rollback1.txt")
        test_file2 = Path("test_rollback2.txt")
        test_file1.write_text("content1\n", encoding="utf-8")
        test_file2.write_text("content2\n", encoding="utf-8")

        # First apply a valid patch
        diff1 = f"""--- {test_file1}
+++ {test_file1}
@@ -1 +1 @@
-content1
+modified1
--- {test_file2}
+++ {test_file2}
@@ -1 +1 @@
-content2
+modified2
"""
        apply_patch(diff1)
        assert "modified1" in test_file1.read_text()
        assert "modified2" in test_file2.read_text()

        # Then try an invalid patch that should fail
        bad_diff = f"""--- {test_file1}
+++ {test_file1}
@@ -1 +1 @@
-wrong_content
+should_fail
"""
        try:
            apply_patch(bad_diff)
            pytest.fail("Should have raised PatchError")
        except PatchError:
            # Files should be rolled back to previous state
            assert "modified1" in test_file1.read_text()
            assert "modified2" in test_file2.read_text()

        try:
            test_file1.unlink(missing_ok=True)
            test_file2.unlink(missing_ok=True)
            bak1 = test_file1.with_suffix(test_file1.suffix + ".bak")
            bak2 = test_file2.with_suffix(test_file2.suffix + ".bak")
            bak1.unlink(missing_ok=True)
            bak2.unlink(missing_ok=True)
        except Exception:
            pass

    def test_apply_rejects_binary_file(self):
        """Should reject patching binary files."""
        test_file = Path("test_binary.bin")
        # Create a file with null bytes (binary)
        test_file.write_bytes(b"binary\x00content")

        diff = f"""--- {test_file}
+++ {test_file}
@@ -1 +1 @@
-binary
+text
"""
        try:
            with pytest.raises(PatchError, match="binary"):
                apply_patch(diff)
        finally:
            test_file.unlink(missing_ok=True)

    def test_apply_rejects_large_file(self):
        """Should reject patching files exceeding size limit."""
        test_file = Path("test_large.txt")
        # Create a large file (>1MB)
        test_file.write_text("x" * (2 * 1024 * 1024), encoding="utf-8")

        diff = f"""--- {test_file}
+++ {test_file}
@@ -1 +1 @@
-{test_file.read_text()[:10]}
+modified
"""
        try:
            with pytest.raises(PatchError, match="too large"):
                apply_patch(diff)
        finally:
            test_file.unlink(missing_ok=True)

    def test_apply_with_hash_verification(self):
        """Should verify file hash if provided."""
        import hashlib
        test_file = Path("test_hash.txt")
        test_file.write_text("original\n", encoding="utf-8")
        original_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()

        diff = f"""--- {test_file}
+++ {test_file}
@@ -1 +1 @@
-original
+modified
"""
        try:
            # Should succeed with correct hash
            apply_patch(diff, expected_hashes={str(test_file): original_hash})

            # Modify file
            test_file.write_text("tampered\n", encoding="utf-8")

            # Should fail with wrong hash
            with pytest.raises(PatchError, match="Hash mismatch"):
                apply_patch(diff, expected_hashes={str(test_file): original_hash})
        finally:
            test_file.unlink(missing_ok=True)
            bak = test_file.with_suffix(test_file.suffix + ".bak")
            bak.unlink(missing_ok=True)


class TestIsBinary:
    """Tests for _is_binary function."""

    def test_is_binary_null_bytes(self):
        """Should detect null bytes as binary."""
        test_file = Path("test_null.bin")
        test_file.write_bytes(b"text\x00more")
        try:
            assert _is_binary(test_file)
        finally:
            test_file.unlink(missing_ok=True)

    def test_is_binary_text_file(self):
        """Should detect text file as not binary."""
        test_file = Path("test_text.txt")
        test_file.write_text("text content\n", encoding="utf-8")
        try:
            assert not _is_binary(test_file)
        finally:
            test_file.unlink(missing_ok=True)

    def test_is_binary_unicode_error(self):
        """Should detect encoding errors as binary."""
        test_file = Path("test_encoding.bin")
        # Write bytes that can't be decoded as UTF-8
        test_file.write_bytes(b"\xff\xfe\x00\x01")
        try:
            assert _is_binary(test_file)
        finally:
            test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
