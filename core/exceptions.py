class LocalCodeAgentError(Exception):
    """Base exception for Local Code Agent."""
    pass


class LLMError(LocalCodeAgentError):
    """Errors related to LLM communication or availability."""
    pass


class PlanningError(LocalCodeAgentError):
    """Errors during the planning phase."""
    pass


class ExecutionError(LocalCodeAgentError):
    """Errors during the execution phase."""
    pass


class ConfigurationError(LocalCodeAgentError):
    """Errors related to configuration."""
    pass


class SecurityError(LocalCodeAgentError):
    """Errors related to security violations (e.g. path traversal)."""
    pass
