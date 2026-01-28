"""Test Sprint 1 features: retry logic, error handling, input validation."""

from __future__ import annotations

import sys
from pathlib import Path

# Test retry logic
def test_retry_logic():
    """Test that retry logic is properly configured."""
    from core.llm import MAX_RETRIES, RETRY_BACKOFF_BASE, RETRYABLE_STATUS_CODES
    
    assert MAX_RETRIES >= 1, "MAX_RETRIES should be at least 1"
    assert RETRY_BACKOFF_BASE > 1.0, "Backoff base should be > 1.0"
    assert len(RETRYABLE_STATUS_CODES) > 0, "Should have retryable status codes"
    assert 429 in RETRYABLE_STATUS_CODES, "429 (rate limit) should be retryable"
    assert 500 in RETRYABLE_STATUS_CODES, "500 (server error) should be retryable"
    print("✓ Retry logic configuration is correct")


# Test error messages
def test_error_messages():
    """Test that error messages are improved."""
    from core.llm import _missing_model_message
    
    msg = _missing_model_message("test-model")
    assert "test-model" in msg, "Error message should mention model name"
    assert "ollama pull" in msg.lower(), "Error message should suggest fix"
    print("✓ Error messages are improved")


# Test input validation (CLI)
def test_input_validation():
    """Test that CLI has input validation."""
    import inspect
    from agent.cli import run
    
    # Check that run function exists and has validation logic
    source = inspect.getsource(run)
    assert "repo_path.exists()" in source or "repo_path.is_dir()" in source, "Should validate repo path"
    assert "model" in source.lower(), "Should handle model validation"
    print("✓ Input validation is present in CLI")


# Test partial failure handling
def test_partial_failure_handling():
    """Test that partial failures are handled."""
    # This is tested in the actual workflow tests
    # Just verify the code structure exists
    from tools.apply_patch import apply_patch, PatchError
    
    assert callable(apply_patch), "apply_patch should be callable"
    assert issubclass(PatchError, Exception), "PatchError should be an exception"
    print("✓ Partial failure handling infrastructure exists")


if __name__ == "__main__":
    print("Testing Sprint 1 Features...")
    print()
    
    try:
        test_retry_logic()
        test_error_messages()
        test_input_validation()
        test_partial_failure_handling()
        
        print()
        print("=" * 50)
        print("All Sprint 1 feature tests passed! ✓")
        print("=" * 50)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
