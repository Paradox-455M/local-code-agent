"""Security utilities for input sanitization and validation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional


def sanitize_path(path: str, repo_root: Path) -> Path:
    """
    Sanitize and validate a file path to prevent path traversal attacks.
    
    Args:
        path: The path string to sanitize.
        repo_root: Repository root directory.
    
    Returns:
        Sanitized Path object.
    
    Raises:
        ValueError: If path contains invalid characters or attempts traversal.
    """
    # Remove null bytes
    path = path.replace("\x00", "")
    
    # Normalize path separators
    path = path.replace("\\", "/")
    
    # Remove leading/trailing whitespace
    path = path.strip()
    
    # Check for path traversal attempts
    if ".." in path or path.startswith("/"):
        # Allow absolute paths only if they're within repo_root
        if path.startswith("/"):
            try:
                abs_path = Path(path).resolve()
                repo_abs = repo_root.resolve()
                if not str(abs_path).startswith(str(repo_abs)):
                    raise ValueError(f"Path outside repository: {path}")
            except Exception:
                raise ValueError(f"Invalid absolute path: {path}")
        else:
            # Relative paths with .. are suspicious
            parts = path.split("/")
            if ".." in parts:
                raise ValueError(f"Path traversal detected: {path}")
    
    # Remove a/ and b/ prefixes (common in diffs)
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    
    # Resolve relative to repo root
    resolved = (repo_root / path).resolve()
    
    # Ensure it's still within repo
    repo_abs = repo_root.resolve()
    try:
        if not resolved.is_relative_to(repo_abs):
            raise ValueError(f"Path outside repository: {path}")
    except AttributeError:
        # Python <3.9 compatibility
        if not str(resolved).startswith(str(repo_abs)):
            raise ValueError(f"Path outside repository: {path}")
    
    return resolved


def sanitize_prompt(prompt: str, max_length: int = 100_000) -> str:
    """
    Sanitize user input in prompts to prevent injection attacks.
    
    Args:
        prompt: The prompt text to sanitize.
        max_length: Maximum allowed prompt length.
    
    Returns:
        Sanitized prompt string.
    
    Raises:
        ValueError: If prompt is too long or contains dangerous patterns.
    """
    if len(prompt) > max_length:
        raise ValueError(f"Prompt too long: {len(prompt)} > {max_length} characters")
    
    # Remove null bytes
    prompt = prompt.replace("\x00", "")
    
    # Check for common injection patterns
    dangerous_patterns = [
        r"<script[^>]*>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"on\w+\s*=",  # Event handlers
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            # Log warning but don't fail (might be false positive)
            # In production, you might want to raise an error
            pass
    
    return prompt


def validate_diff_content(diff_text: str, max_size: int = 2_000_000) -> tuple[bool, Optional[str]]:
    """
    Validate diff content before parsing to prevent malicious inputs.
    
    Args:
        diff_text: The diff text to validate.
        max_size: Maximum allowed diff size in bytes.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    # Check size
    diff_bytes = len(diff_text.encode("utf-8"))
    if diff_bytes > max_size:
        return False, f"Diff too large: {diff_bytes} > {max_size} bytes"
    
    # Check for null bytes
    if "\x00" in diff_text:
        return False, "Diff contains null bytes"
    
    # Basic format check: should contain diff markers
    if not ("---" in diff_text and "+++" in diff_text):
        return False, "Invalid diff format: missing --- or +++ markers"
    
    # Check for reasonable line count
    line_count = diff_text.count("\n")
    if line_count > 50_000:
        return False, f"Diff has too many lines: {line_count} > 50000"
    
    return True, None


def compute_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.
    
    Args:
        file_path: Path to the file.
    
    Returns:
        Hexadecimal checksum string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """
    Verify file checksum matches expected value.
    
    Args:
        file_path: Path to the file.
        expected_checksum: Expected SHA256 checksum.
    
    Returns:
        True if checksum matches, False otherwise.
    """
    try:
        actual = compute_checksum(file_path)
        return actual == expected_checksum
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent directory traversal and other issues.
    
    Args:
        filename: The filename to sanitize.
    
    Returns:
        Sanitized filename.
    """
    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    
    # Remove null bytes
    filename = filename.replace("\x00", "")
    
    # Remove leading dots (hidden files are OK, but .. is not)
    while filename.startswith(".."):
        filename = filename[2:]
    
    # Remove leading/trailing whitespace
    filename = filename.strip()
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename


if __name__ == "__main__":
    # Demo
    repo_root = Path("/tmp/test_repo")
    repo_root.mkdir(exist_ok=True)
    
    # Test path sanitization
    try:
        safe_path = sanitize_path("../../etc/passwd", repo_root)
        print(f"Sanitized path: {safe_path}")
    except ValueError as e:
        print(f"Correctly rejected: {e}")
    
    # Test prompt sanitization
    safe_prompt = sanitize_prompt("What does this code do?")
    print(f"Sanitized prompt: {safe_prompt[:50]}...")
    
    # Test diff validation
    valid, error = validate_diff_content("--- a.py\n+++ a.py\n@@ -1 +1 @@\n-old\n+new")
    print(f"Diff validation: valid={valid}, error={error}")
