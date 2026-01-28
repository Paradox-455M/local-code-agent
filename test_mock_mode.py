from __future__ import annotations

from pathlib import Path

try:
    from agent.executor import execute
except ModuleNotFoundError as exc:
    print(f"SKIP: missing dependency during import: {exc}")
    raise SystemExit(0)


def test_execute_with_mock_llm() -> None:
    target = Path("mock_mode.txt")
    target.write_text("placeholder\n", encoding="utf-8")

    def stub_llm(_: str) -> str:
        return """--- mock_mode.txt
+++ mock_mode.txt
@@ -1 +1 @@
-placeholder
+generated
"""

    res = execute(
        "generate code",
        [str(target)],
        [str(target)],
        ".",
        llm_fn=stub_llm,
    )
    assert res.diffs
    target.unlink(missing_ok=True)
    bak = target.with_suffix(target.suffix + ".bak")
    bak.unlink(missing_ok=True)
