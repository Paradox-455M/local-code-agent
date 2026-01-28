"""Reasoning and explanation system - Claude Code level."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level for decisions."""
    
    VERY_HIGH = "very_high"  # >90%
    HIGH = "high"  # 70-90%
    MEDIUM = "medium"  # 50-70%
    LOW = "low"  # 30-50%
    VERY_LOW = "very_low"  # <30%


@dataclass
class Decision:
    """A decision made by the agent."""
    
    decision_type: str  # 'file_selection', 'approach', 'strategy', etc.
    value: Any  # The decision value
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Explanation of why
    alternatives: List[Any] = field(default_factory=list)
    factors: Dict[str, float] = field(default_factory=dict)  # Factors that influenced decision


@dataclass
class Explanation:
    """Human-readable explanation."""
    
    summary: str
    reasoning: str
    confidence: ConfidenceLevel
    alternatives: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class Reasoner:
    """Provides reasoning and explanations like Claude Code."""
    
    def explain_file_selection(
        self,
        task: str,
        selected_files: List[str],
        scores: Dict[str, float],
        all_candidates: Optional[List[str]] = None,
    ) -> Explanation:
        """
        Explain why certain files were selected.
        
        Args:
            task: The task description.
            selected_files: Files that were selected.
            scores: Scores for each file.
            all_candidates: All candidate files considered.
        
        Returns:
            Explanation of file selection.
        """
        if not selected_files:
            return Explanation(
                summary="No files selected",
                reasoning="No files matched the task criteria.",
                confidence=ConfidenceLevel.LOW,
            )
        
        # Calculate average confidence
        avg_score = sum(scores.get(f, 0.0) for f in selected_files) / len(selected_files)
        max_possible_score = 100.0  # Assuming max score is around 100
        confidence_ratio = min(1.0, avg_score / max_possible_score)
        
        if confidence_ratio > 0.9:
            confidence = ConfidenceLevel.VERY_HIGH
        elif confidence_ratio > 0.7:
            confidence = ConfidenceLevel.HIGH
        elif confidence_ratio > 0.5:
            confidence = ConfidenceLevel.MEDIUM
        elif confidence_ratio > 0.3:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.VERY_LOW
        
        # Build reasoning
        reasoning_parts = []
        
        # Top file reasoning
        if selected_files:
            top_file = selected_files[0]
            top_score = scores.get(top_file, 0.0)
            reasoning_parts.append(
                f"Selected '{top_file}' (score: {top_score:.1f}) as the primary file because it "
                f"most closely matches the task requirements."
            )
        
        # Multiple files reasoning
        if len(selected_files) > 1:
            reasoning_parts.append(
                f"Included {len(selected_files) - 1} additional related file(s) to provide "
                f"necessary context for the task."
            )
        
        # Alternative files
        alternatives = []
        if all_candidates:
            # Find files that were close but not selected
            candidate_scores = [(f, scores.get(f, 0.0)) for f in all_candidates if f not in selected_files]
            candidate_scores.sort(key=lambda x: -x[1])
            alternatives = [f"{f} (score: {s:.1f})" for f, s in candidate_scores[:3]]
        
        reasoning = " ".join(reasoning_parts)
        
        # Suggestions
        suggestions = []
        if confidence in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW):
            suggestions.append("Consider providing more specific file paths or symbols to improve selection accuracy.")
        if len(selected_files) == 1 and len(all_candidates or []) > 5:
            suggestions.append("Multiple files might be relevant. Consider reviewing related files.")
        
        return Explanation(
            summary=f"Selected {len(selected_files)} file(s) based on task analysis",
            reasoning=reasoning,
            confidence=confidence,
            alternatives=alternatives,
            suggestions=suggestions,
        )
    
    def explain_approach(
        self,
        task: str,
        approach: str,
        reasoning: str,
        confidence: float = 0.7,
    ) -> Explanation:
        """
        Explain why a particular approach was chosen.
        
        Args:
            task: The task description.
            approach: The approach chosen.
            reasoning: Reasoning for the approach.
            confidence: Confidence level (0.0 to 1.0).
        
        Returns:
            Explanation of the approach.
        """
        if confidence > 0.9:
            conf_level = ConfidenceLevel.VERY_HIGH
        elif confidence > 0.7:
            conf_level = ConfidenceLevel.HIGH
        elif confidence > 0.5:
            conf_level = ConfidenceLevel.MEDIUM
        elif confidence > 0.3:
            conf_level = ConfidenceLevel.LOW
        else:
            conf_level = ConfidenceLevel.VERY_LOW
        
        alternatives = []
        suggestions = []
        
        if conf_level in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW):
            suggestions.append("Consider providing more details about the desired approach.")
            alternatives.append("Alternative approaches might be more suitable. Please clarify requirements.")
        
        return Explanation(
            summary=f"Using {approach} approach",
            reasoning=reasoning,
            confidence=conf_level,
            alternatives=alternatives,
            suggestions=suggestions,
        )
    
    def explain_diff_strategy(
        self,
        task: str,
        diffs: List[Dict[str, str]],
        strategy: str,
    ) -> Explanation:
        """
        Explain why a particular diff strategy was used.
        
        Args:
            task: The task description.
            diffs: Generated diffs.
            strategy: Strategy used (e.g., 'llm', 'deterministic', 'hybrid').
        
        Returns:
            Explanation of diff strategy.
        """
        num_files = len(diffs)
        
        reasoning = f"Generated {num_files} diff(s) using {strategy} strategy. "
        
        if strategy == "llm":
            reasoning += "Used LLM-based generation for complex code changes requiring understanding of context."
        elif strategy == "deterministic":
            reasoning += "Used pattern-based replacement for straightforward text modifications."
        else:
            reasoning += "Used hybrid approach combining pattern matching and LLM generation."
        
        suggestions = []
        if num_files == 0:
            suggestions.append("No changes generated. Consider clarifying the task or providing more context.")
        elif num_files > 5:
            suggestions.append("Many files affected. Review carefully to ensure all changes are correct.")
        
        return Explanation(
            summary=f"Generated {num_files} diff(s) using {strategy}",
            reasoning=reasoning,
            confidence=ConfidenceLevel.MEDIUM,
            suggestions=suggestions,
        )
    
    def suggest_alternatives(
        self,
        current_plan: Dict[str, Any],
        task: str,
    ) -> List[str]:
        """
        Suggest alternative approaches to the current plan.
        
        Args:
            current_plan: Current execution plan.
            task: Task description.
        
        Returns:
            List of alternative suggestions.
        """
        alternatives = []
        
        files_to_modify = current_plan.get("files_to_modify", [])
        if len(files_to_modify) == 0:
            alternatives.append("Consider specifying target files explicitly if the task requires file modifications.")
        
        if len(files_to_modify) > 5:
            alternatives.append("Many files affected. Consider breaking this into smaller, incremental changes.")
        
        mode = current_plan.get("mode", "auto")
        if mode == "auto":
            alternatives.append("Consider specifying a mode (bugfix/refactor/feature) for more targeted changes.")
        
        return alternatives
    
    def ask_clarification(
        self,
        task: str,
        ambiguity: str,
    ) -> str:
        """
        Generate a clarifying question when task is ambiguous.
        
        Args:
            task: The ambiguous task.
            ambiguity: What is ambiguous.
        
        Returns:
            Clarifying question.
        """
        questions = {
            "file": "Which file(s) should I modify?",
            "function": "Which function(s) should I modify?",
            "approach": "What approach would you prefer?",
            "scope": "What is the scope of this change?",
            "style": "Should I follow any specific coding style or patterns?",
        }
        
        question = questions.get(ambiguity.lower(), "Could you provide more details?")
        return f"I need clarification: {question}"
    
    def assess_confidence(
        self,
        plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ConfidenceLevel:
        """
        Assess confidence in a plan.
        
        Args:
            plan: Execution plan.
            context: Context information.
        
        Returns:
            Confidence level.
        """
        confidence_factors = []
        
        # File selection confidence
        files_to_modify = plan.get("files_to_modify", [])
        if files_to_modify:
            confidence_factors.append(0.3)  # Files specified
        else:
            confidence_factors.append(0.1)  # No files specified
        
        # Context availability
        has_context = context.get("has_context", False)
        if has_context:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # Task clarity
        task = plan.get("goal", "")
        if len(task.split()) > 5:  # More detailed task
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # Mode specification
        mode = plan.get("mode", "auto")
        if mode != "auto":
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.05)
        
        avg_confidence = sum(confidence_factors)
        
        if avg_confidence > 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif avg_confidence > 0.6:
            return ConfidenceLevel.HIGH
        elif avg_confidence > 0.4:
            return ConfidenceLevel.MEDIUM
        elif avg_confidence > 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


def create_reasoner() -> Reasoner:
    """Create a reasoner instance."""
    return Reasoner()


if __name__ == "__main__":
    # Demo
    reasoner = create_reasoner()
    
    # Explain file selection
    explanation = reasoner.explain_file_selection(
        task="fix bug in main.py",
        selected_files=["main.py", "utils.py"],
        scores={"main.py": 95.0, "utils.py": 75.0},
        all_candidates=["main.py", "utils.py", "config.py", "test_main.py"],
    )
    
    print(f"Summary: {explanation.summary}")
    print(f"Reasoning: {explanation.reasoning}")
    print(f"Confidence: {explanation.confidence.value}")
    print(f"Alternatives: {explanation.alternatives}")
