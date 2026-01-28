"""Agentic loop with self-correction and iterative improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
from enum import Enum
import subprocess
import re

try:
    from agent.refiner import create_refiner, CodeRefiner
    REFINER_AVAILABLE = True
except ImportError:
    REFINER_AVAILABLE = False
    CodeRefiner = None  # type: ignore
    create_refiner = None  # type: ignore


class LoopStatus(Enum):
    """Status of the agentic loop."""
    
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"
    USER_CANCELLED = "user_cancelled"


@dataclass
class LoopResult:
    """Result of agentic loop execution."""
    
    status: LoopStatus
    iterations: int
    final_result: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    feedback_history: List[str] = field(default_factory=list)


class AgenticLoop:
    """Self-correcting execution loop that iterates until success."""
    
    def __init__(
        self,
        execute_fn: Callable[[str], Any],
        verify_fn: Optional[Callable[[Any], bool]] = None,
        max_iterations: int = 3,
        auto_fix_syntax: bool = True,
        auto_fix_lint: bool = True,
        auto_fix_tests: bool = True,
        enable_refinement: bool = True,
    ):
        """
        Initialize agentic loop.
        
        Args:
            execute_fn: Function that executes a task and returns a result.
            verify_fn: Optional function to verify result quality. Returns True if good.
            max_iterations: Maximum number of iterations before giving up.
            auto_fix_syntax: Whether to auto-fix syntax errors.
            auto_fix_lint: Whether to auto-fix linting errors.
            auto_fix_tests: Whether to auto-fix test failures.
            enable_refinement: Whether to use incremental refinement.
        """
        self.execute_fn = execute_fn
        self.verify_fn = verify_fn or self._default_verify
        self.max_iterations = max_iterations
        self.auto_fix_syntax = auto_fix_syntax
        self.auto_fix_lint = auto_fix_lint
        self.auto_fix_tests = auto_fix_tests
        self.enable_refinement = enable_refinement and REFINER_AVAILABLE
        self.refiner: Optional[CodeRefiner] = None
        if self.enable_refinement:
            try:
                self.refiner = create_refiner()
            except Exception:
                self.enable_refinement = False
    
    def _default_verify(self, result: Any) -> bool:
        """Default verification: check if result has errors."""
        if isinstance(result, dict):
            # Check for common error indicators
            if result.get("error"):
                return False
            if result.get("errors"):
                return False
            if result.get("success") is False:
                return False
        return True
    
    def execute_with_retry(self, task: str, initial_context: Optional[Dict[str, Any]] = None) -> LoopResult:
        """
        Execute task with retry and self-correction.
        
        Args:
            task: The task to execute.
            initial_context: Optional initial context for the task.
        
        Returns:
            LoopResult with execution status and history.
        """
        current_task = task
        context = initial_context or {}
        errors = []
        feedback_history = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Execute task
            try:
                result = self.execute_fn(current_task)
            except Exception as e:
                errors.append(f"Iteration {iteration}: Execution failed: {e}")
                feedback = self._analyze_failure(None, str(e))
                feedback_history.append(feedback)
                
                if iteration < self.max_iterations:
                    current_task = self._refine_task(current_task, feedback, context)
                    continue
                else:
                    return LoopResult(
                        status=LoopStatus.FAILED,
                        iterations=iteration,
                        errors=errors,
                        feedback_history=feedback_history,
                    )
            
            # Verify result
            if self.verify_fn(result):
                return LoopResult(
                    status=LoopStatus.SUCCESS,
                    iterations=iteration,
                    final_result=result,
                    feedback_history=feedback_history,
                )
            
            # Analyze failure
            feedback = self._analyze_failure(result, None)
            feedback_history.append(feedback)
            errors.append(f"Iteration {iteration}: {feedback}")
            
            # Refine task based on feedback
            if iteration < self.max_iterations:
                # Use refiner if available for better refinement
                if self.enable_refinement and self.refiner and isinstance(result, dict):
                    # Try to refine the generated code if we have it
                    if "diffs" in result and result["diffs"]:
                        # Extract code from diffs and refine
                        refined_task = self._refine_task_with_refiner(current_task, feedback, result)
                        if refined_task:
                            current_task = refined_task
                        else:
                            current_task = self._refine_task(current_task, feedback, context)
                    else:
                        current_task = self._refine_task(current_task, feedback, context)
                else:
                    current_task = self._refine_task(current_task, feedback, context)
            else:
                return LoopResult(
                    status=LoopStatus.MAX_ITERATIONS,
                    iterations=iteration,
                    final_result=result,
                    errors=errors,
                    feedback_history=feedback_history,
                )
        
        return LoopResult(
            status=LoopStatus.MAX_ITERATIONS,
            iterations=self.max_iterations,
            errors=errors,
            feedback_history=feedback_history,
        )
    
    def _analyze_failure(self, result: Optional[Any], error: Optional[str] = None) -> str:
        """
        Analyze why execution failed.
        
        Args:
            result: The result object (may contain error info).
            error: Optional error message.
        
        Returns:
            Analysis of the failure.
        """
        if error:
            # Analyze error message
            if "SyntaxError" in error or "syntax error" in error.lower():
                return "Syntax error detected. Need to fix code syntax."
            if "ImportError" in error or "ModuleNotFoundError" in error:
                return "Import error detected. Need to check imports."
            if "NameError" in error:
                return "Name error detected. Variable or function not defined."
            return f"Execution error: {error}"
        
        if result is None:
            return "Execution returned no result."
        
        if isinstance(result, dict):
            if result.get("syntax_error"):
                return "Syntax error in generated code."
            if result.get("lint_errors"):
                return f"Linting errors: {result.get('lint_errors')}"
            if result.get("test_failures"):
                return f"Test failures: {result.get('test_failures')}"
            if result.get("error"):
                return f"Error: {result.get('error')}"
        
        return "Result verification failed."
    
    def _refine_task(self, task: str, feedback: str, context: Dict[str, Any]) -> str:
        """
        Refine task based on feedback.
        
        Args:
            task: Original task.
            feedback: Feedback from failure analysis.
            context: Current context.
        
        Returns:
            Refined task.
        """
        # Simple refinement: append feedback to task
        refined = f"{task}\n\nPrevious attempt failed: {feedback}. Please fix the issues."
        
        # Add specific instructions based on feedback type
        if "syntax" in feedback.lower():
            refined += "\n\nFocus on fixing syntax errors. Ensure all brackets, parentheses, and quotes are properly closed."
        elif "lint" in feedback.lower():
            refined += "\n\nFocus on fixing linting errors. Follow code style guidelines."
        elif "test" in feedback.lower():
            refined += "\n\nFocus on fixing test failures. Ensure code matches test expectations."
        elif "import" in feedback.lower():
            refined += "\n\nFocus on fixing import errors. Check that all required modules are imported."
        
        return refined
    
    def _refine_task_with_refiner(
        self,
        task: str,
        feedback: str,
        result: Dict[str, Any],
    ) -> Optional[str]:
        """
        Refine task using code refiner for better results.
        
        Args:
            task: Original task.
            feedback: Feedback from failure analysis.
            result: Previous execution result.
        
        Returns:
            Refined task or None if refinement not applicable.
        """
        if not self.refiner:
            return None
        
        # Extract code from diffs if available
        diffs = result.get("diffs", [])
        if not diffs:
            return None
        
        # Try to refine based on feedback
        # Build a more specific task with feedback
        refined_task = f"{task}\n\nFeedback: {feedback}\n\nPlease refine the code to address the feedback."
        
        return refined_task
    
    def verify_with_tests(self, result: Any, test_command: str = "pytest -q") -> bool:
        """
        Verify result by running tests.
        
        Args:
            result: Result to verify.
            test_command: Command to run tests.
        
        Returns:
            True if tests pass.
        """
        try:
            proc = subprocess.run(
                test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            return proc.returncode == 0
        except Exception:
            return False
    
    def verify_with_lint(self, result: Any, lint_command: str = "ruff check .") -> bool:
        """
        Verify result by running linter.
        
        Args:
            result: Result to verify.
            lint_command: Command to run linter.
        
        Returns:
            True if linting passes.
        """
        try:
            proc = subprocess.run(
                lint_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return proc.returncode == 0
        except Exception:
            return False


def create_agentic_loop(
    execute_fn: Callable[[str], Any],
    verify_fn: Optional[Callable[[Any], bool]] = None,
    **kwargs
) -> AgenticLoop:
    """
    Create an agentic loop with default settings.
    
    Args:
        execute_fn: Function that executes a task.
        verify_fn: Optional verification function.
        **kwargs: Additional arguments for AgenticLoop.
    
    Returns:
        Configured AgenticLoop instance.
    """
    return AgenticLoop(execute_fn, verify_fn, **kwargs)


if __name__ == "__main__":
    # Demo
    def mock_execute(task: str):
        """Mock execution function."""
        print(f"Executing: {task}")
        # Simulate failure on first attempt
        if "fix" not in task.lower():
            return {"error": "Syntax error", "success": False}
        return {"success": True, "result": "Task completed"}
    
    loop = AgenticLoop(mock_execute, max_iterations=3)
    result = loop.execute_with_retry("add new feature")
    
    print(f"\nStatus: {result.status}")
    print(f"Iterations: {result.iterations}")
    print(f"Errors: {result.errors}")
    print(f"Feedback: {result.feedback_history}")
