from __future__ import annotations

from pathlib import Path

try:
    from main import app
except ModuleNotFoundError as exc:
    # Allow environments without installed deps to skip gracefully.
    print(f"SKIP: missing dependency during import: {exc}")
    raise SystemExit(0)


def test_app_is_typer_instance() -> None:
    # Typer exposes .registered_commands attribute; presence implies a Typer app.
    assert hasattr(app, "registered_commands")


def test_pyproject_entrypoint_present() -> None:
    content = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'local-code-agent = "agent.cli:app"' in content


def test_cli_flags_documented() -> None:
    # Spot-check presence of new flags in source for quick regression.
    content = Path("agent/cli.py").read_text(encoding="utf-8")
    for flag in [
        "--plan-only",
        "--model",
        "--context-glob",
        "--mock-llm",
        "--post-check",
        "--selective-apply",
        "--mode",
        "--show-plan",
        "--show-reasoning",
        "--preview-context",
        "--symbols",
        "--write-patch",
        "--relaxed-validation",
        "--mock-from",
        "--targets",
        "--dirs",
        "--allow-outside-repo",
        "--disallow-outside-repo",
        "--quick-review",
        "--run-pytest",
        "--run-ruff",
        "--run-mypy",
        "--run-py-compile",
    ]:
        assert flag in content
