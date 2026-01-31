"""Configuration file support for local-code-agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import tomli
except ImportError:
    tomli = None  # type: ignore

CONFIG_FILENAME = ".local-code-agent.toml"
CONFIG_PROFILES = ["dev", "prod", "test"]


def find_config_file(repo_root: str) -> Optional[Path]:
    """
    Find the configuration file in the repository.
    
    Args:
        repo_root: Repository root directory.
    
    Returns:
        Path to config file if found, None otherwise.
    """
    repo_path = Path(repo_root)
    config_path = repo_path / CONFIG_FILENAME
    
    if config_path.exists() and config_path.is_file():
        return config_path
    
    return None


def load_config(repo_root: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        repo_root: Repository root directory.
        profile: Optional profile name (dev, prod, test).
    
    Returns:
        Configuration dictionary.
    """
    if tomli is None:
        # tomli not available, return empty config
        return {}
    
    config_path = find_config_file(repo_root)
    if not config_path:
        return {}
    
    try:
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)
        
        # If profile specified, merge profile config
        if profile and profile in config_data.get("profiles", {}):
            profile_config = config_data["profiles"][profile]
            # Merge with base config (profile overrides base)
            base_config = {k: v for k, v in config_data.items() if k != "profiles"}
            return {**base_config, **profile_config}
        
        return config_data
    except Exception:
        return {}


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a configuration value, supporting nested keys.
    
    Args:
        config: Configuration dictionary.
        key: Key path (e.g., "agent.max_files").
        default: Default value if not found.
    
    Returns:
        Configuration value or default.
    """
    keys = key.split(".")
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def validate_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate configuration file.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        Tuple of (is_valid, errors).
    """
    errors = []
    
    # Validate agent section
    if "agent" in config:
        agent_config = config["agent"]
        if "max_files" in agent_config:
            if not isinstance(agent_config["max_files"], int) or agent_config["max_files"] < 1:
                errors.append("agent.max_files must be a positive integer")
        if "always_plan" in agent_config:
            if not isinstance(agent_config["always_plan"], bool):
                errors.append("agent.always_plan must be a boolean")
        
        if "model" in agent_config:
            if not isinstance(agent_config["model"], str):
                errors.append("agent.model must be a string")
    
    # Validate llm section
    if "llm" in config:
        llm_config = config["llm"]
        if "temperature" in llm_config:
            temp = llm_config["temperature"]
            if not isinstance(temp, (int, float)) or not (0 <= temp <= 2):
                errors.append("llm.temperature must be between 0 and 2")
    
    return len(errors) == 0, errors


def create_default_config(repo_root: str) -> Path:
    """
    Create a default configuration file.
    
    Args:
        repo_root: Repository root directory.
    
    Returns:
        Path to created config file.
    
    Raises:
        ImportError: If tomli is not available.
    """
    if tomli is None:
        raise ImportError(
            "tomli is required for config file support. "
            "Install it with: pip install tomli"
        )
    
    config_path = Path(repo_root) / CONFIG_FILENAME
    
    default_config = """# Local Code Agent Configuration

[agent]
# Maximum number of files to include in context
max_files = 5

# Always show plan and require approval before proceeding
# always_plan = false

# Default model to use (overrides LCA_MODEL env var)
# model = "llama3.2"

# Default mode (auto, bugfix, feature, refactor, general)
# mode = "auto"

[llm]
# Temperature for LLM calls (0.0-2.0)
temperature = 0.1

# Maximum retries for LLM calls
max_retries = 3

[prompts]
# Use enhanced prompts (task-specific)
use_enhanced = true

# Use few-shot examples
use_few_shot = true

# Use chain-of-thought for complex tasks
use_chain_of_thought = true

[validation]
# Auto-correct diffs before applying
auto_correct = true

# Run syntax checks before applying
pre_validate = true

[profiles.dev]
# Development profile overrides
# max_files = 10

[profiles.prod]
# Production profile overrides
# max_files = 3
"""
    
    config_path.write_text(default_config, encoding="utf-8")
    return config_path


if __name__ == "__main__":
    # Demo
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = create_default_config(tmpdir)
        print(f"Created config at: {config_path}")
        
        config = load_config(tmpdir)
        print(f"Loaded config: {config}")
        
        max_files = get_config_value(config, "agent.max_files", 5)
        print(f"max_files: {max_files}")
        
        is_valid, errors = validate_config(config)
        print(f"Valid: {is_valid}, Errors: {errors}")
