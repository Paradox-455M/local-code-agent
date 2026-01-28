from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

try:
    from core.config_file import load_config, get_config_value
except ImportError:
    # Fallback if config_file not available
    def load_config(*args, **kwargs):  # type: ignore
        return {}
    def get_config_value(*args, **kwargs):  # type: ignore
        return None

DEFAULT_ALLOWED_EXTS = [".py", ".md", ".txt", ".json", ".yaml", ".yml"]
DEFAULT_DENYLIST = [
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "coverage",
]


def _parse_list(env_value: str | None, default: List[str]) -> List[str]:
    if not env_value:
        return default
    # Accept comma-separated values; keep simple, no trimming of empty tokens
    return [item.strip() for item in env_value.split(",") if item.strip()]


@dataclass
class Config:
    # Default model (override with LCA_MODEL env var or config file)
    model: str = os.getenv("LCA_MODEL", "qwen3-coder:latest")
    repo_root: Path = Path(os.getenv("LCA_REPO_ROOT", Path.cwd()))
    max_plan_files: int = int(os.getenv("LCA_MAX_PLAN_FILES", 50))
    allowed_exts: List[str] = field(
        default_factory=lambda: _parse_list(os.getenv("LCA_ALLOWED_EXTS"), DEFAULT_ALLOWED_EXTS)
    )
    denylist_paths: List[str] = field(
        default_factory=lambda: _parse_list(os.getenv("LCA_DENYLIST_PATHS"), DEFAULT_DENYLIST)
    )
    # LLM backend configuration
    llm_backend: str = os.getenv("LCA_LLM_BACKEND", "ollama")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # Git integration
    auto_commit: bool = os.getenv("LCA_AUTO_COMMIT", "false").lower() == "true"
    auto_branch: bool = os.getenv("LCA_AUTO_BRANCH", "false").lower() == "true"
    # Performance: skip KG/semantic for faster runs (env: LCA_FAST_MODE, LCA_USE_KNOWLEDGE_GRAPH)
    fast_mode: bool = os.getenv("LCA_FAST_MODE", "false").lower() in ("true", "1", "yes")
    use_knowledge_graph: bool = os.getenv("LCA_USE_KNOWLEDGE_GRAPH", "true").lower() in ("true", "1", "yes")
    # Summary caching and LLM summaries
    summary_mode: str = os.getenv("LCA_SUMMARY_MODE", "struct")  # struct|llm|hybrid
    summary_max_bytes: int = int(os.getenv("LCA_SUMMARY_MAX_BYTES", "4000"))
    summary_chunk_lines: int = int(os.getenv("LCA_SUMMARY_CHUNK_LINES", "120"))
    summary_max_chunks: int = int(os.getenv("LCA_SUMMARY_MAX_CHUNKS", "8"))
    graph_cache_ttl: int = int(os.getenv("LCA_GRAPH_CACHE_TTL", "300"))
    _config_file_data: Optional[dict] = None
    
    def __post_init__(self):
        """Load config file after initialization."""
        self._load_config_file()
    
    def _load_config_file(self, profile: Optional[str] = None):
        """Load configuration from file."""
        try:
            if self.fast_mode:
                self.use_knowledge_graph = False
            repo_str = str(self.repo_root)
            file_config = load_config(repo_str, profile=profile)
            if file_config:
                self._config_file_data = file_config

                # Override with config file values (if not set via env)
                if "agent" in file_config:
                    agent_cfg = file_config["agent"]
                    if "model" in agent_cfg and not os.getenv("LCA_MODEL"):
                        self.model = agent_cfg["model"]
                    if "max_files" in agent_cfg:
                        self.max_plan_files = agent_cfg["max_files"]
        except Exception:
            # Silently fail if config file not available
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from config file."""
        if self._config_file_data:
            return get_config_value(self._config_file_data, key, default)
        return default


config = Config()
