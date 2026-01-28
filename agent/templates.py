"""Template system for common coding tasks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class Template:
    """A task template."""
    
    name: str
    description: str
    task_pattern: str  # Regex pattern to match tasks
    prompt_template: str  # Template with {placeholders}
    parameters: List[str]  # List of parameter names
    examples: List[Dict[str, str]] = None  # Example usages
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []
    
    def matches(self, task: str) -> bool:
        """Check if a task matches this template."""
        return bool(re.search(self.task_pattern, task, re.IGNORECASE))
    
    def extract_parameters(self, task: str) -> Dict[str, str]:
        """Extract parameters from a task."""
        match = re.search(self.task_pattern, task, re.IGNORECASE)
        if not match:
            return {}
        
        params = {}
        for param in self.parameters:
            # Try to extract from named groups
            if param in match.groupdict():
                params[param] = match.group(param)
            else:
                # Try to extract from position
                params[param] = match.group(0) if match.groups() else task
        
        return params
    
    def generate_prompt(self, task: str, context: Optional[str] = None) -> str:
        """Generate a prompt from the template."""
        params = self.extract_parameters(task)
        
        # Fill in template
        prompt = self.prompt_template
        for key, value in params.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        
        # Add context if provided
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}"
        
        return prompt


class TemplateLibrary:
    """Library of task templates."""
    
    def __init__(self):
        self.templates: List[Template] = []
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default templates."""
        # Add function template
        self.add_template(Template(
            name="add_function",
            description="Add a new function to a file",
            task_pattern=r"(?:add|create|implement)\s+(?:a\s+)?function\s+(?:called|named)?\s*(?P<function_name>\w+)(?:\s+that\s+(?P<description>.+?))?(?:\s+in\s+(?P<file>\S+))?",
            prompt_template="Add a function named {function_name} to the codebase.\n\nDescription: {description}\n\nFile: {file}",
            parameters=["function_name", "description", "file"],
            examples=[
                {"task": "add function calculate_total", "params": {"function_name": "calculate_total"}},
                {"task": "create function parse_config in config.py", "params": {"function_name": "parse_config", "file": "config.py"}},
            ],
        ))
        
        # Fix bug template
        self.add_template(Template(
            name="fix_bug",
            description="Fix a bug or error",
            task_pattern=r"fix\s+(?:the\s+)?(?:bug|error|issue)\s+(?:in\s+)?(?P<file>\S+)?(?:\s+(?:where|that)\s+(?P<description>.+?))?",
            prompt_template="Fix the bug in {file}.\n\nIssue: {description}\n\nAnalyze the code, identify the root cause, and apply a minimal fix.",
            parameters=["file", "description"],
            examples=[
                {"task": "fix the bug in main.py", "params": {"file": "main.py"}},
                {"task": "fix error where division by zero occurs", "params": {"description": "division by zero occurs"}},
            ],
        ))
        
        # Refactor template
        self.add_template(Template(
            name="refactor",
            description="Refactor code",
            task_pattern=r"refactor\s+(?P<target>.+?)(?:\s+to\s+(?P<goal>.+?))?",
            prompt_template="Refactor {target}.\n\nGoal: {goal}\n\nImprove code structure without changing behavior.",
            parameters=["target", "goal"],
            examples=[
                {"task": "refactor the executor module", "params": {"target": "the executor module"}},
                {"task": "refactor calculate_total to use generators", "params": {"target": "calculate_total", "goal": "use generators"}},
            ],
        ))
        
        # Add test template
        self.add_template(Template(
            name="add_test",
            description="Add tests for code",
            task_pattern=r"(?:add|create|write)\s+(?:a\s+)?test\s+(?:for|of)\s+(?P<target>.+?)",
            prompt_template="Add comprehensive tests for {target}.\n\nInclude edge cases and error handling.",
            parameters=["target"],
            examples=[
                {"task": "add test for calculate_total", "params": {"target": "calculate_total"}},
                {"task": "create tests for the executor module", "params": {"target": "the executor module"}},
            ],
        ))
        
        # Update documentation template
        self.add_template(Template(
            name="update_docs",
            description="Update documentation",
            task_pattern=r"(?:update|improve|fix)\s+(?:the\s+)?(?:docs?|documentation)\s+(?:for\s+)?(?P<target>.+?)?",
            prompt_template="Update documentation for {target}.\n\nEnsure it's clear, accurate, and complete.",
            parameters=["target"],
            examples=[
                {"task": "update docs for the API", "params": {"target": "the API"}},
                {"task": "improve documentation", "params": {"target": ""}},
            ],
        ))
    
    def add_template(self, template: Template) -> None:
        """Add a template to the library."""
        self.templates.append(template)
    
    def find_matching_template(self, task: str) -> Optional[Template]:
        """Find a template that matches the task."""
        for template in self.templates:
            if template.matches(task):
                return template
        return None
    
    def load_from_file(self, file_path: Path) -> None:
        """Load templates from a JSON file."""
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            for item in data.get("templates", []):
                template = Template(
                    name=item["name"],
                    description=item["description"],
                    task_pattern=item["task_pattern"],
                    prompt_template=item["prompt_template"],
                    parameters=item.get("parameters", []),
                    examples=item.get("examples", []),
                )
                self.add_template(template)
        except Exception as e:
            raise ValueError(f"Failed to load templates from {file_path}: {e}") from e
    
    def save_to_file(self, file_path: Path) -> None:
        """Save templates to a JSON file."""
        data = {
            "templates": [
                {
                    "name": t.name,
                    "description": t.description,
                    "task_pattern": t.task_pattern,
                    "prompt_template": t.prompt_template,
                    "parameters": t.parameters,
                    "examples": t.examples,
                }
                for t in self.templates
            ]
        }
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# Global template library instance
_template_library: Optional[TemplateLibrary] = None


def get_template_library() -> TemplateLibrary:
    """Get the global template library instance."""
    global _template_library
    if _template_library is None:
        _template_library = TemplateLibrary()
    return _template_library


def find_template_for_task(task: str) -> Optional[Template]:
    """Find a template for a given task."""
    return get_template_library().find_matching_template(task)


if __name__ == "__main__":
    # Demo
    library = TemplateLibrary()
    
    test_tasks = [
        "add function calculate_total",
        "fix the bug in main.py",
        "refactor the executor",
        "add test for calculate_total",
        "update docs for the API",
    ]
    
    for task in test_tasks:
        template = library.find_matching_template(task)
        if template:
            print(f"Task: {task}")
            print(f"  Template: {template.name}")
            print(f"  Parameters: {template.extract_parameters(task)}")
            print()
