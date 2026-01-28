from __future__ import annotations

from pathlib import Path

from git import Repo, InvalidGitRepositoryError

try:
    from core.config import config
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.config import config  # type: ignore


def git_status() -> str:
    """
    Return the git status text for the repo root.
    """
    try:
        repo = Repo(config.repo_root, search_parent_directories=True)
    except InvalidGitRepositoryError as exc:  # pragma: no cover
        raise RuntimeError("Not a git repository") from exc
    return repo.git.status()


if __name__ == "__main__":
    print(git_status())
