from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    from core.config import config
    from core.security import sanitize_path, validate_diff_content
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.config import config  # type: ignore
    try:
        from core.security import sanitize_path, validate_diff_content
    except ImportError:
        sanitize_path = None  # type: ignore
        validate_diff_content = None  # type: ignore

HUNK_RE = re.compile(r"@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@")
MAX_PATCH_BYTES = 2_000_000  # 2MB safeguard
MAX_PATCH_LINES = 5000  # limit per file to avoid huge patches


@dataclass
class FilePatch:
    path: Path
    hunks: List[List[str]]


class PatchError(RuntimeError):
    pass


def _ensure_within_repo(path: Path) -> Path:
    root = config.repo_root.resolve()
    # Use security sanitization if available
    if sanitize_path is not None:
        try:
            # Convert Path to string for sanitization
            path_str = str(path)
            if path.is_absolute():
                # For absolute paths, make relative to repo root
                try:
                    path_str = str(path.relative_to(root))
                except ValueError:
                    pass
            return sanitize_path(path_str, root)
        except ValueError as e:
            raise PatchError(str(e)) from e
    
    # Fallback to original logic
    # Interpret relative patch targets as relative to the repo root, not the
    # current working directory (which may be a subdir).
    candidate = path if path.is_absolute() else (root / path)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise PatchError(f"Path outside repo root: {resolved}")
    return resolved


def _normalize_path(raw: str) -> str:
    token = raw.split("\t")[0].strip()
    if token.startswith("a/") or token.startswith("b/"):
        token = token[2:]
    return token


def _parse_diff(diff: str, allow_outside_repo: bool = False) -> List[FilePatch]:
    # Validate diff content before parsing (skip for empty diffs)
    if diff.strip() and validate_diff_content is not None:
        is_valid, error = validate_diff_content(diff)
        if not is_valid:
            raise PatchError(f"Invalid diff content: {error}")
    lines = diff.splitlines()
    if len(lines) > MAX_PATCH_LINES:
        raise PatchError("Diff too large to apply safely")
    patches: List[FilePatch] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if not line.startswith("--- "):
            i += 1
            continue

        if i + 1 >= n or not lines[i + 1].startswith("+++ "):
            raise PatchError("Invalid diff header; expected +++ line after ---")

        target_raw = _normalize_path(lines[i + 1][4:])
        target_path = Path(target_raw)
        if not allow_outside_repo:
            target_path = _ensure_within_repo(target_path)
        else:
            # If relative, still resolve from repo root for consistency.
            # If absolute, allow outside of repo root.
            target_path = _ensure_within_repo(target_path) if not target_path.is_absolute() else target_path.resolve()
        i += 2

        hunks: List[List[str]] = []
        current: List[str] | None = None

        while i < n and not lines[i].startswith("--- "):
            cur_line = lines[i]
            if cur_line.startswith("@@"):
                if current is not None:
                    hunks.append(current)
                current = [cur_line]
            else:
                if current is None:
                    raise PatchError("Hunk data found before header")
                current.append(cur_line)
            i += 1

        if current:
            hunks.append(current)
        if not hunks:
            raise PatchError("No hunks found in diff for file")

        patches.append(FilePatch(path=target_path, hunks=hunks))

    if not patches:
        raise PatchError("No file patches detected in diff")
    return patches


def _apply_hunks(orig_lines: list[str], hunks: list[list[str]]) -> list[str]:
    out: list[str] = []
    idx = 0  # 0-based index in orig_lines

    for hunk in hunks:
        header = hunk[0].strip()
        match = HUNK_RE.match(header)
        if not match:
            raise PatchError(f"Invalid hunk header: {header}")
        # Unified diff uses 1-based line numbers, except new files which often
        # use old_start=0 (e.g. @@ -0,0 +1,3 @@).
        old_start_raw = int(match.group("old_start"))
        old_start = 0 if old_start_raw == 0 else (old_start_raw - 1)
        old_len = int(match.group("old_len") or "1")

        out.extend(orig_lines[idx:old_start])
        idx = old_start

        for line in hunk[1:]:
            # Standard unified diffs may include this marker line; ignore it.
            if line == r"\ No newline at end of file":
                continue
            if not line:
                raise PatchError("Empty line inside hunk; expected diff tag prefix")
            tag, content = line[0], line[1:]
            if tag == " ":
                if idx >= len(orig_lines) or orig_lines[idx] != content:
                    raise PatchError("Context mismatch while applying patch")
                out.append(content)
                idx += 1
            elif tag == "-":
                if idx >= len(orig_lines) or orig_lines[idx] != content:
                    raise PatchError("Deletion mismatch while applying patch")
                idx += 1
            elif tag == "+":
                out.append(content)
            else:
                raise PatchError(f"Unknown hunk tag: {tag}")

        expected_idx = old_start + old_len
        if idx != expected_idx and old_len != 0:
            raise PatchError("Hunk application drifted; aborting")

    out.extend(orig_lines[idx:])
    return out


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(1024)
            if b"\0" in chunk:
                return True
            chunk.decode("utf-8")
    except UnicodeDecodeError:
        return True
    except OSError as exc:
        raise PatchError(f"Unable to read file: {path}") from exc
    return False


def apply_patch(
    diff: str,
    expected_hashes: dict[str, str] | None = None,
    keep_backup: bool = True,
    allow_outside_repo: bool = False,
) -> List[str]:
    """
    Apply a unified diff to files within the repo (multi-file supported).

    - Validates target paths are inside repo root.
    - Only patches existing, non-binary files below size limit.
    - Creates a .bak backup before writing (kept on success).
    - Rolls back all changes if any file fails to apply.
    Returns: list of modified file paths.
    """
    patches = _parse_diff(diff, allow_outside_repo=allow_outside_repo)
    modified: List[str] = []
    backups: List[Tuple[Path, Path]] = []

    try:
        for file_patch in patches:
            target_path = file_patch.path
            exists = target_path.exists() and target_path.is_file()

            if exists and target_path.stat().st_size > MAX_PATCH_BYTES:
                raise PatchError(f"Target too large to patch safely: {target_path}")

            if exists and _is_binary(target_path):
                raise PatchError(f"Refusing to patch binary file: {target_path}")

            if expected_hashes and exists:
                import hashlib

                current_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
                # Allow callers to key expected_hashes by either absolute or
                # repo-relative path.
                expected = expected_hashes.get(str(target_path))
                if expected is None:
                    try:
                        rel = target_path.relative_to(config.repo_root.resolve()).as_posix()
                    except Exception:
                        rel = None
                    if rel:
                        expected = expected_hashes.get(rel)
                if expected and expected != current_hash:
                    raise PatchError(f"Hash mismatch for {target_path}; refusing to apply.")

            if exists:
                original_text = target_path.read_text(encoding="utf-8", errors="replace")
                orig_lines = original_text.splitlines()
                trailing_newline = original_text.endswith("\n")
            else:
                original_text = ""
                orig_lines = []
                trailing_newline = False

            patched_lines = _apply_hunks(orig_lines, file_patch.hunks)

            if exists:
                backup_path = target_path.with_suffix(target_path.suffix + ".bak")
                shutil.copy2(target_path, backup_path)
                backups.append((target_path, backup_path))

            new_text = "\n".join(patched_lines)
            if trailing_newline or patched_lines:
                new_text += "\n"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(new_text, encoding="utf-8")
            modified.append(str(target_path.resolve()))

    except Exception as exc:
        for target_path, backup_path in backups:
            if backup_path.exists():
                shutil.copy2(backup_path, target_path)
        raise PatchError(str(exc)) from exc

    if not keep_backup:
        for _, backup_path in backups:
            if backup_path.exists():
                backup_path.unlink()

    return modified


if __name__ == "__main__":
    test_path = Path("apply_patch_demo.txt")
    test_path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    demo_diff = """--- apply_patch_demo.txt
+++ apply_patch_demo.txt
@@ -1,3 +1,4 @@
 line1
-line2
+line2 changed
 line3
+line4 added
"""
    try:
        modified = apply_patch(demo_diff)
        print("Patched files:", modified)
        print("New content:")
        print(test_path.read_text())
    finally:
        if test_path.exists():
            test_path.unlink()
        bak = test_path.with_suffix(test_path.suffix + ".bak")
        if bak.exists():
            bak.unlink()
