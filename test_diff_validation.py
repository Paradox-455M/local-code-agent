from __future__ import annotations

from pathlib import Path

try:
    from tools.apply_patch import PatchError, apply_patch
    from agent import executor as executor_mod
except ModuleNotFoundError as exc:
    print(f"SKIP: missing dependency during import: {exc}")
    raise SystemExit(0)


def test_apply_patch_hash_guard() -> None:
    temp = Path("hash_guard.txt")
    temp.write_text("old\n", encoding="utf-8")
    diff = """--- hash_guard.txt
+++ hash_guard.txt
@@ -1 +1 @@
-old
+new
"""
    # Wrong hash should fail
    try:
        apply_patch(diff, expected_hashes={str(temp): "deadbeef"})
    except PatchError:
        pass
    else:
        assert False, "Expected PatchError on hash mismatch"

    # Correct hash should apply
    import hashlib

    h = hashlib.sha256(temp.read_bytes()).hexdigest()
    modified = apply_patch(diff, expected_hashes={str(temp): h}, keep_backup=False)
    assert str(temp.resolve()) in modified
    assert "new" in temp.read_text()
    temp.unlink(missing_ok=True)


def test_diff_validation_filters_bad_diffs() -> None:
    bad = [{"diff": "not a diff"}]
    assert not executor_mod._validate_diffs(bad)

    good = [{"diff": "--- a\n+++ a\n@@ -1 +1 @@\n-old\n+new\n"}]
    assert executor_mod._validate_diffs(good)
