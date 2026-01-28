from __future__ import annotations

import difflib


def unified_diff(old: str, new: str, path: str) -> str:
    """
    Return a unified diff string between old and new content for a given path.
    """
    # Use non-keepend splitting so the diff is "standard unified diff" without
    # extra blank lines between entries. (apply_patch expects this format.)
    old_lines = old.splitlines(keepends=False)
    new_lines = new.splitlines(keepends=False)
    diff_iter = difflib.unified_diff(
        old_lines, new_lines, fromfile=path, tofile=path, lineterm=""
    )
    return "\n".join(diff_iter)


if __name__ == "__main__":
    before = "line1\nline2\nline3\n"
    after = "line1\nline2 changed\nline3\nline4 added\n"
    print(unified_diff(before, after, "example.txt"))
