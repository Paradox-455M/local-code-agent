from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class Review:
    risks: List[str]
    edge_cases: List[str]
    explanation: str
    suggested_tests: List[str]


def review_diffs(task: str, diffs: List[Dict[str, str]]) -> Review:
    if not diffs:
        return Review(
            risks=["No diffs provided; nothing to verify."],
            edge_cases=[],
            explanation=(
                "No changes were proposed, so nothing was applied. "
                "Likely reason: executor lacked enough context or safe instructions."
            ),
            suggested_tests=["Clarify the intended change and re-run."],
        )

    touched_files = [d["path"] for d in diffs if "path" in d]
    risks = [
        "Verify patched files compile and preserve functionality.",
        "Ensure backups (.bak) are cleaned after apply if not needed.",
    ]
    edge_cases = ["Large files or binary content should not be patched blindly."]
    explanation = f"Task: {task}. Diffs prepared for {', '.join(touched_files)}."
    suggested_tests = ["python -m py_compile " + " ".join(touched_files)]

    return Review(risks=risks, edge_cases=edge_cases, explanation=explanation, suggested_tests=suggested_tests)


def to_json(review: Review) -> str:
    """Serialize the Review object to a JSON string."""
    return json.dumps(asdict(review), indent=2)


if __name__ == "__main__":
    sample = review_diffs("demo", [{"path": "core/config.py", "diff": ""}])
    print(to_json(sample))
