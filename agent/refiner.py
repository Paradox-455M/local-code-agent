"""Incremental code refinement - Claude Code level iterative improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
import difflib


@dataclass
class RefinementRequest:
    """Request for code refinement."""
    
    original_code: str
    feedback: str
    target_lines: Optional[tuple[int, int]] = None  # (start_line, end_line) if refining specific part
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Result of code refinement."""
    
    refined_code: str
    changes_made: List[str]  # Description of changes
    confidence: float  # 0.0 to 1.0
    preserved_parts: List[str] = field(default_factory=list)  # Parts that were preserved


class CodeRefiner:
    """Iterative code refinement like Claude Code."""
    
    def __init__(
        self,
        refine_fn: Optional[Callable[[str, str], str]] = None,
        preserve_unchanged: bool = True,
    ):
        """
        Initialize code refiner.
        
        Args:
            refine_fn: Optional function to refine code (uses LLM if None).
            preserve_unchanged: Whether to preserve unchanged parts of code.
        """
        self.refine_fn = refine_fn
        self.preserve_unchanged = preserve_unchanged
    
    def refine(
        self,
        code: str,
        feedback: str,
        target_lines: Optional[tuple[int, int]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RefinementResult:
        """
        Refine code based on feedback.
        
        Args:
            code: Original code.
            feedback: Feedback on what to improve.
            target_lines: Optional (start, end) line range to refine.
            context: Optional context for refinement.
        
        Returns:
            RefinementResult with refined code.
        """
        if target_lines:
            # Refine specific part
            return self._refine_partial(code, feedback, target_lines, context or {})
        else:
            # Refine entire code
            return self._refine_full(code, feedback, context or {})
    
    def _refine_full(self, code: str, feedback: str, context: Dict[str, Any]) -> RefinementResult:
        """Refine entire code."""
        if self.refine_fn:
            refined = self.refine_fn(code, feedback)
        else:
            # Use LLM for refinement
            refined = self._refine_with_llm(code, feedback, context)
        
        # Analyze changes
        changes = self._analyze_changes(code, refined)
        preserved = self._find_preserved_parts(code, refined)
        
        # Calculate confidence based on change size
        confidence = self._calculate_confidence(code, refined, changes)
        
        return RefinementResult(
            refined_code=refined,
            changes_made=changes,
            confidence=confidence,
            preserved_parts=preserved,
        )
    
    def _refine_partial(
        self,
        code: str,
        feedback: str,
        target_lines: tuple[int, int],
        context: Dict[str, Any],
    ) -> RefinementResult:
        """Refine specific part of code."""
        lines = code.splitlines()
        start_line, end_line = target_lines
        
        # Extract target section
        target_section = "\n".join(lines[start_line - 1:end_line])
        before_section = "\n".join(lines[:start_line - 1])
        after_section = "\n".join(lines[end_line:])
        
        # Refine target section
        if self.refine_fn:
            refined_section = self.refine_fn(target_section, feedback)
        else:
            refined_section = self._refine_with_llm(target_section, feedback, context)
        
        # Reconstruct code
        refined_code = "\n".join([
            before_section,
            refined_section,
            after_section,
        ]).strip()
        
        # Analyze changes
        changes = [f"Refined lines {start_line}-{end_line}: {feedback}"]
        preserved = [
            f"Lines 1-{start_line - 1}: preserved",
            f"Lines {end_line + 1}-{len(lines)}: preserved",
        ]
        
        confidence = self._calculate_confidence(target_section, refined_section, changes)
        
        return RefinementResult(
            refined_code=refined_code,
            changes_made=changes,
            confidence=confidence,
            preserved_parts=preserved,
        )
    
    def _refine_with_llm(self, code: str, feedback: str, context: Dict[str, Any]) -> str:
        """Refine code using LLM."""
        try:
            from core.llm import ask
            
            prompt = f"""Refine the following code based on the feedback.

Code:
```python
{code}
```

Feedback: {feedback}

Requirements:
- Fix the issues mentioned in feedback
- Maintain code style and structure
- Preserve functionality that's not being changed
- Return only the refined code, no explanations

Refined code:"""
            
            refined = ask(prompt)
            
            # Extract code block if LLM wrapped it
            if "```" in refined:
                lines = refined.split("\n")
                code_lines = []
                in_code = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code = not in_code
                        continue
                    if in_code:
                        code_lines.append(line)
                if code_lines:
                    refined = "\n".join(code_lines)
            
            return refined.strip()
        except Exception:
            # Fallback: return original code
            return code
    
    def _analyze_changes(self, original: str, refined: str) -> List[str]:
        """Analyze what changed between original and refined code."""
        changes = []
        
        original_lines = original.splitlines()
        refined_lines = refined.splitlines()
        
        diff = list(difflib.unified_diff(original_lines, refined_lines, lineterm=""))
        
        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
        
        if added > 0:
            changes.append(f"Added {added} line(s)")
        if removed > 0:
            changes.append(f"Removed {removed} line(s)")
        
        # Detect type of changes
        if any("def " in line for line in refined_lines if line not in original_lines):
            changes.append("Function definition modified")
        if any("class " in line for line in refined_lines if line not in original_lines):
            changes.append("Class definition modified")
        
        return changes if changes else ["Code refined"]
    
    def _find_preserved_parts(self, original: str, refined: str) -> List[str]:
        """Find parts of code that were preserved."""
        if not self.preserve_unchanged:
            return []
        
        original_lines = original.splitlines()
        refined_lines = refined.splitlines()
        
        # Find common subsequences
        matcher = difflib.SequenceMatcher(None, original_lines, refined_lines)
        preserved = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal" and (i2 - i1) > 3:  # Preserved blocks of 3+ lines
                preserved.append(f"Lines {i1 + 1}-{i2}: preserved ({i2 - i1} lines)")
        
        return preserved
    
    def _calculate_confidence(
        self,
        original: str,
        refined: str,
        changes: List[str],
    ) -> float:
        """Calculate confidence in refinement."""
        if original == refined:
            return 0.0  # No changes made
        
        # Higher confidence if:
        # - Small, focused changes
        # - Structure preserved
        # - Syntax likely valid
        
        original_lines = len(original.splitlines())
        refined_lines = len(refined.splitlines())
        
        # Confidence based on change size (smaller changes = higher confidence)
        change_ratio = abs(refined_lines - original_lines) / max(original_lines, 1)
        size_confidence = max(0.0, 1.0 - change_ratio)
        
        # Check if structure preserved
        structure_preserved = self._check_structure_preserved(original, refined)
        structure_confidence = 0.8 if structure_preserved else 0.5
        
        # Combined confidence
        confidence = (size_confidence * 0.6 + structure_confidence * 0.4)
        
        return min(1.0, max(0.0, confidence))
    
    def _check_structure_preserved(self, original: str, refined: str) -> bool:
        """Check if code structure is preserved."""
        # Simple check: same number of functions/classes
        original_funcs = original.count("def ")
        refined_funcs = refined.count("def ")
        original_classes = original.count("class ")
        refined_classes = refined.count("class ")
        
        return (
            abs(original_funcs - refined_funcs) <= 1 and
            abs(original_classes - refined_classes) <= 1
        )
    
    def refine_incrementally(
        self,
        code: str,
        feedback_list: List[str],
        max_iterations: int = 3,
    ) -> RefinementResult:
        """
        Refine code incrementally based on multiple feedback items.
        
        Args:
            code: Original code.
            feedback_list: List of feedback items to address.
            max_iterations: Maximum refinement iterations.
        
        Returns:
            Final refinement result.
        """
        current_code = code
        all_changes = []
        all_preserved = []
        
        for i, feedback in enumerate(feedback_list[:max_iterations], 1):
            result = self.refine(current_code, feedback)
            current_code = result.refined_code
            all_changes.extend([f"Iteration {i}: {c}" for c in result.changes_made])
            all_preserved.extend(result.preserved_parts)
        
        # Calculate overall confidence
        overall_confidence = min(1.0, sum(
            self._calculate_confidence(code, current_code, all_changes)
            for _ in range(len(feedback_list))
        ) / len(feedback_list) if feedback_list else 0.5)
        
        return RefinementResult(
            refined_code=current_code,
            changes_made=all_changes,
            confidence=overall_confidence,
            preserved_parts=list(set(all_preserved)),
        )


def create_refiner(refine_fn: Optional[Callable[[str, str], str]] = None) -> CodeRefiner:
    """
    Create a code refiner instance.
    
    Args:
        refine_fn: Optional refinement function (uses LLM if None).
    
    Returns:
        CodeRefiner instance.
    """
    return CodeRefiner(refine_fn=refine_fn)


if __name__ == "__main__":
    # Demo
    code = """def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total"""
    
    feedback = "Add error handling for empty items list"
    
    refiner = create_refiner()
    result = refiner.refine(code, feedback)
    
    print("Original:")
    print(code)
    print("\nRefined:")
    print(result.refined_code)
    print(f"\nChanges: {result.changes_made}")
    print(f"Confidence: {result.confidence:.2f}")
