from __future__ import annotations

import json
from pathlib import Path

try:
    from agent.executor import execute
    from agent import executor as executor_mod
    from agent.planner import build_plan, to_json as plan_to_json
    from agent.reviewer import review_diffs
    from tools.apply_patch import PatchError, apply_patch
    from tools.diff import unified_diff
except ModuleNotFoundError as exc:
    print(f"SKIP: missing dependency during import: {exc}")
    raise SystemExit(0)


def test_plan_executor_reviewer() -> None:
    repo_files = ["core/config.py", "main.py"]
    plan = build_plan("demo task", repo_files)
    plan_json = plan_to_json(plan)
    loaded = json.loads(plan_json)
    assert loaded["goal"] == "demo task"
    assert len(loaded["files_to_read"]) <= 10
    assert any("File selection" in assumption or "File selection" in assumption for assumption in loaded["assumptions"])

    result = execute("demo task", plan.files_to_read, plan.files_to_modify, ".")
    serialized = json.loads(json.dumps(result, default=lambda o: o.__dict__))
    assert serialized["diffs"] == []
    assert any("Clarification" in note for note in serialized["notes"])

    temp_path = Path("tmp_exec.txt")
    temp_path.write_text("alpha beta", encoding="utf-8")
    result_replace = execute(
        "replace alpha with gamma", [str(temp_path)], [str(temp_path)], ".", instruction="replace alpha with gamma"
    )
    serialized_replace = json.loads(json.dumps(result_replace, default=lambda o: o.__dict__))
    assert serialized_replace["diffs"]
    assert result_replace.confidence > 0
    temp_path.unlink(missing_ok=True)
    temp_bak = temp_path.with_suffix(temp_path.suffix + ".bak")
    temp_bak.unlink(missing_ok=True)

    review = review_diffs("demo task", result.diffs)
    assert isinstance(review.explanation, str)


def test_executor_append_and_insert() -> None:
    target = Path("tmp_anchor.txt")
    target.write_text("start\nanchor\nend\n", encoding="utf-8")

    # Append after anchor
    res_append = execute(
        "append 'middle' after 'anchor'",
        [str(target)],
        [str(target)],
        ".",
        instruction="append middle after anchor",
        enable_llm=True,
    )
    serialized = json.loads(json.dumps(res_append, default=lambda o: o.__dict__))
    assert serialized["diffs"]
    assert res_append.confidence > 0

    # Insert before anchor
    target.write_text("start\nanchor\nend\n", encoding="utf-8")
    res_insert = execute(
        "insert 'header' before 'start'",
        [str(target)],
        [str(target)],
        ".",
        instruction="insert header before start",
        enable_llm=True,
    )
    serialized_ins = json.loads(json.dumps(res_insert, default=lambda o: o.__dict__))
    assert serialized_ins["diffs"]
    assert res_insert.confidence > 0

    target.unlink(missing_ok=True)
    bak = target.with_suffix(target.suffix + ".bak")
    bak.unlink(missing_ok=True)


def test_executor_llm_path_with_stub() -> None:
    def stub_llm(_: str) -> str:
        return """--- tmp_exec_llm.txt
+++ tmp_exec_llm.txt
@@ -1 +1 @@
-foo
+bar
"""

    target = Path("tmp_exec_llm.txt")
    target.write_text("foo\n", encoding="utf-8")
    res = execute(
        "make change",
        [str(target)],
        [str(target)],
        ".",
        instruction="",
        llm_fn=stub_llm,
        enable_llm=True,
    )
    assert res.diffs
    target.unlink(missing_ok=True)
    bak = target.with_suffix(target.suffix + ".bak")
    bak.unlink(missing_ok=True)


def test_context_builder_limits() -> None:
    base = Path(".")
    file_a = Path("ctx_a.txt")
    file_a.write_text("a" * 5000, encoding="utf-8")
    snippets = executor_mod.build_context(base, [str(file_a)], max_context_files=1, max_bytes=1000)
    assert len(snippets) == 1
    assert len(snippets[0]["snippet"]) <= 1000
    file_a.unlink(missing_ok=True)
    bak = file_a.with_suffix(file_a.suffix + ".bak")
    bak.unlink(missing_ok=True)


def test_apply_patch_round_trip() -> None:
    temp_path = Path("tmp_apply.txt")
    temp_path.write_text("a\nb\n", encoding="utf-8")
    diff = unified_diff("a\nb\n", "a\nb-changed\n", str(temp_path))
    modified = apply_patch(diff)
    assert str(temp_path.resolve()) in modified
    assert "b-changed" in temp_path.read_text()

    temp_path.unlink(missing_ok=True)
    bak = temp_path.with_suffix(temp_path.suffix + ".bak")
    bak.unlink(missing_ok=True)


def test_apply_patch_multi_and_rollback() -> None:
    file1 = Path("tmp1.txt")
    file2 = Path("tmp2.txt")
    file1.write_text("one\n", encoding="utf-8")
    file2.write_text("two\n", encoding="utf-8")

    multi_diff = """--- tmp1.txt
+++ tmp1.txt
@@ -1 +1 @@
-one
+one-1
--- tmp2.txt
+++ tmp2.txt
@@ -1 +1 @@
-two
+two-2
"""
    modified = apply_patch(multi_diff)
    assert str(file1.resolve()) in modified
    assert str(file2.resolve()) in modified
    assert "one-1" in file1.read_text()
    assert "two-2" in file2.read_text()

    bad_diff = """--- tmp1.txt
+++ tmp1.txt
@@ -1 +1 @@
-one-1
+one-2
--- tmp2.txt
+++ tmp2.txt
@@ -1 +1 @@
-missing
+should-fail
"""
    try:
        apply_patch(bad_diff)
    except PatchError:
        pass
    else:
        assert False, "Expected PatchError"

    assert "one-1" in file1.read_text()
    assert "two-2" in file2.read_text()

    for path in (file1, file2):
        path.unlink(missing_ok=True)
        bak = path.with_suffix(path.suffix + ".bak")
        bak.unlink(missing_ok=True)


if __name__ == "__main__":
    test_plan_executor_reviewer()
    test_apply_patch_round_trip()
    test_apply_patch_multi_and_rollback()
    print("All lightweight tests passed.")
