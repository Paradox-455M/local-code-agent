from __future__ import annotations

import sys
from pathlib import Path

try:
    from core.config import config
except ImportError:
    # Allow running as a standalone script by adding repo root to sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.config import config  # type: ignore


def _validate_path(path: Path, repo_root: Path) -> Path:
    resolved_root = repo_root.resolve()
    candidate = path if path.is_absolute() else resolved_root / path
    resolved_candidate = candidate.resolve()

    if not resolved_candidate.is_file():
        raise FileNotFoundError(f"File not found: {resolved_candidate}")

    if not resolved_candidate.is_relative_to(resolved_root):
        raise PermissionError("Access denied: path is outside the repository root")

    return resolved_candidate


def read_file(path: str) -> str:
    """
    Read a text file within the repository using UTF-8 with replacement.

    Rejects attempts to read outside the repository root.
    """
    target_path = _validate_path(Path(path), config.repo_root)
    try:
        with target_path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as exc:
        raise RuntimeError(f"Unable to read file: {exc}") from exc


if __name__ == "__main__":
    # Simple CLI check: print length and the first 200 characters of this file.
    content = read_file(__file__)
    preview = content[:200].replace("\n", "\\n")
    print(f"Bytes read: {len(content.encode('utf-8'))}")
    print(f"Preview: {preview}")
