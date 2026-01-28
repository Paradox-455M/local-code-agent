"""Enhanced prompt engineering for different task types."""

from typing import Iterator, Optional
from core.llm import ask_stream
from agent.task_classifier import TaskType, classify_task


def enhance_task_prompt(task: str) -> str:
    """Enhance a task prompt with structured formatting."""
    return (
        "You are a Senior Technical Product Manager. Refine the following coding task "
        "into a structured technical requirement for a developer.\n"
        "The goal is to clarify the user's intent and provide specific technical context "
        "that will help an AI coding agent locate the right files and implement the solution.\n\n"
        f"Input Task: {task}\n\n"
        "Please restructure this into:\n"
        "1. **Summary**: A one-sentence technical summary.\n"
        "2. **Objectives**: Specific goals to achieve.\n"
        "3. **Technical Context**: Relevant keywords, potential file types, or architectural patterns to look for.\n\n"
        "Do not write code. Do not solve the task. Just refine the prompt."
    )


def get_task_type_prompt(task: str, task_type: TaskType, context: Optional[str] = None, conventions: Optional[str] = None) -> str:
    """
    Get a task-specific prompt optimized for the task type.
    
    Args:
        task: The task description.
        task_type: The classified task type.
        context: Optional context about the codebase.
        conventions: Optional project conventions to follow.
    
    Returns:
        Optimized prompt for the task type.
    """
    base_context = context or ""
    
    if task_type == TaskType.FIX:
        return _get_fix_prompt(task, base_context, conventions)
    elif task_type == TaskType.REFACTOR:
        return _get_refactor_prompt(task, base_context, conventions)
    elif task_type == TaskType.CREATE:
        return _get_create_prompt(task, base_context, conventions)
    elif task_type == TaskType.MODIFY:
        return _get_modify_prompt(task, base_context, conventions)
    elif task_type == TaskType.EXPLAIN:
        return _get_explain_prompt(task, base_context)
    else:
        return _get_general_prompt(task, base_context, conventions)


def _get_fix_prompt(task: str, context: str, conventions: Optional[str] = None) -> str:
    """Prompt optimized for bug fixes."""
    conventions_text = f"\n\nCode Style Conventions:\n{conventions}" if conventions else ""
    return f"""You are fixing a bug. Analyze the code carefully and apply a minimal, targeted fix.

Task: {task}

{context}{conventions_text}

Instructions:
1. Identify the root cause of the bug
2. Apply the minimal fix needed
3. Ensure the fix doesn't break existing functionality
4. Follow existing code style and patterns
5. Maintain consistency with project conventions

Generate a unified diff that fixes the issue."""


def _get_refactor_prompt(task: str, context: str, conventions: Optional[str] = None) -> str:
    """Prompt optimized for refactoring."""
    conventions_text = f"\n\nCode Style Conventions:\n{conventions}" if conventions else ""
    return f"""You are refactoring code to improve structure without changing behavior.

Task: {task}

{context}{conventions_text}

Instructions:
1. Maintain existing functionality exactly
2. Improve code organization, readability, or structure
3. Follow SOLID principles where applicable
4. Keep changes focused and incremental
5. Preserve all existing behavior
6. Follow project conventions strictly

Generate a unified diff that refactors the code."""


def _get_create_prompt(task: str, context: str, conventions: Optional[str] = None) -> str:
    """Prompt optimized for creating new code."""
    conventions_text = f"\n\nCode Style Conventions:\n{conventions}" if conventions else ""
    return f"""You are creating new code following existing patterns.

Task: {task}

{context}{conventions_text}

Instructions:
1. Follow existing code style and conventions strictly
2. Match the architecture and patterns used in the codebase
3. Include appropriate docstrings and type hints
4. Add necessary imports
5. Ensure the new code integrates well with existing code
6. Follow project naming conventions and style guidelines

Generate a unified diff that adds the new code."""


def _get_modify_prompt(task: str, context: str, conventions: Optional[str] = None) -> str:
    """Prompt optimized for modifications."""
    conventions_text = f"\n\nCode Style Conventions:\n{conventions}" if conventions else ""
    return f"""You are modifying existing code to add features or make changes.

Task: {task}

{context}{conventions_text}

Instructions:
1. Understand the existing code structure
2. Make targeted changes without breaking existing functionality
3. Follow existing patterns and style strictly
4. Update related code if necessary
5. Maintain backward compatibility when possible
6. Follow project conventions

Generate a unified diff that implements the changes."""


def _get_explain_prompt(task: str, context: str) -> str:
    """Prompt optimized for explanations."""
    return f"""Explain the code clearly and concisely.

Task: {task}

{context}

Provide a clear explanation of:
1. What the code does
2. How it works
3. Key concepts or patterns used
4. Any important details

Do not generate code changes."""


def _get_general_prompt(task: str, context: str, conventions: Optional[str] = None) -> str:
    """General-purpose prompt."""
    conventions_text = f"\n\nCode Style Conventions:\n{conventions}" if conventions else ""
    return f"""You are a coding assistant helping with a development task.

Task: {task}

{context}{conventions_text}

Instructions:
1. Understand the requirements
2. Analyze the existing codebase
3. Generate appropriate code changes
4. Follow existing patterns and style strictly
5. Ensure changes are correct and complete
6. Follow project conventions

Generate a unified diff that implements the task."""


def get_chain_of_thought_prompt(task: str, context: str, is_complex: bool = False) -> str:
    """
    Generate a chain-of-thought prompt for complex tasks.
    
    Args:
        task: The task description.
        context: Codebase context.
        is_complex: Whether this is a complex task.
    
    Returns:
        Chain-of-thought prompt.
    """
    if not is_complex:
        return get_task_type_prompt(task, classify_task(task)[0], context)
    
    return f"""You are solving a complex coding task. Think step by step.

Task: {task}

{context}

Step-by-step approach:
1. **Understand**: What exactly needs to be done?
2. **Analyze**: What files and code sections are relevant?
3. **Plan**: What changes are needed and in what order?
4. **Implement**: Generate the code changes
5. **Verify**: Ensure the changes are correct and complete

Think through each step before generating the diff.

Generate a unified diff that implements the solution."""


# Few-shot examples for common patterns
FEW_SHOT_EXAMPLES = {
    TaskType.FIX: """
Example 1: Fixing a bug
Task: "Fix the division by zero error in calculate_average"
Context: Shows the function with the bug
Diff:
--- calculator.py
+++ calculator.py
@@ -5,7 +5,7 @@
 def calculate_average(numbers):
-    return sum(numbers) / len(numbers)
+    if len(numbers) == 0:
+        return 0
+    return sum(numbers) / len(numbers)
""",
    
    TaskType.REFACTOR: """
Example 1: Extracting a method
Task: "Extract the validation logic into a separate function"
Context: Shows code with inline validation
Diff:
--- user.py
+++ user.py
@@ -10,6 +10,10 @@
+def validate_email(email: str) -> bool:
+    return "@" in email and "." in email.split("@")[1]
+
 def create_user(email: str, name: str):
-    if "@" in email and "." in email.split("@")[1]:
+    if validate_email(email):
         # ... rest of function
""",
    
    TaskType.CREATE: """
Example 1: Creating a new function
Task: "Add a function to format dates"
Context: Shows existing date utilities
Diff:
--- utils.py
+++ utils.py
@@ -5,3 +5,7 @@
+def format_date(date: datetime) -> str:
+    \"\"\"Format a datetime object as YYYY-MM-DD.\"\"\"
+    return date.strftime("%Y-%m-%d")
+
""",
}


def get_prompt_with_examples(task: str, task_type: TaskType, context: str) -> str:
    """
    Get a prompt with few-shot examples for the task type.
    
    Args:
        task: The task description.
        task_type: The task type.
        context: Codebase context.
    
    Returns:
        Prompt with examples.
    """
    base_prompt = get_task_type_prompt(task, task_type, context)
    
    if task_type in FEW_SHOT_EXAMPLES:
        example = FEW_SHOT_EXAMPLES[task_type]
        return f"""{base_prompt}

Here's an example of a similar task:

{example}

Now generate the diff for the current task."""
    
    return base_prompt


def stream_enhanced_task(task: str, model: str | None = None) -> Iterator[str]:
    """
    Streams the enhanced version of the task from the LLM.
    """
    prompt = enhance_task_prompt(task)
    yield from ask_stream(prompt, model=model)
