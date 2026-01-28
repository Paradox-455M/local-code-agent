# Security Considerations

This document outlines security measures implemented in the Local Code Agent.

## Input Sanitization

### File Paths
- **Path Traversal Prevention**: All file paths are sanitized to prevent directory traversal attacks (`../` patterns)
- **Repository Boundary Enforcement**: Paths are validated to ensure they remain within the repository root
- **Null Byte Removal**: Null bytes are removed from paths to prevent injection attacks

### Prompts
- **Length Limits**: Prompts are limited to 100KB by default to prevent resource exhaustion
- **Null Byte Removal**: Null bytes are removed from prompts
- **Injection Pattern Detection**: Common injection patterns are detected (though not blocked to avoid false positives)

### Diffs
- **Size Limits**: Diffs are limited to 2MB and 50,000 lines
- **Format Validation**: Diffs must contain valid `---` and `+++` markers
- **Null Byte Detection**: Null bytes in diffs are rejected

## Rate Limiting

- **LLM API Calls**: Rate limiting prevents abuse of LLM API calls
- **Configurable Limits**: Default is 10 requests per 60 seconds
- **Minimum Interval**: Minimum interval between requests (0.1s default)
- **Thread-Safe**: Rate limiter is thread-safe for concurrent use

## Resource Management

- **Memory Monitoring**: Memory usage is tracked and can be limited
- **Temporary File Cleanup**: Temporary files are registered and cleaned up automatically
- **Resource Limits**: Configurable limits for memory, CPU, and disk usage

## Checksum Verification

- **File Integrity**: SHA256 checksums can be computed and verified for critical files
- **Backup Verification**: Backups created during patch application can be verified

## Security Best Practices

1. **Never Trust User Input**: All user inputs are sanitized before use
2. **Validate Early**: Validation happens as early as possible in the processing pipeline
3. **Fail Securely**: On validation failure, operations fail with clear error messages
4. **Least Privilege**: Operations are restricted to the repository root
5. **Resource Limits**: Limits prevent resource exhaustion attacks

## Known Limitations

- **Prompt Injection**: While patterns are detected, complete prevention is difficult without blocking legitimate use cases
- **Rate Limiting**: Rate limits are per-process, not global across multiple instances
- **File Size**: Large files (>1MB) are skipped during scanning, but not during explicit operations

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly. Do not create public issues for security vulnerabilities.

---

*Last updated: 2026-01-27*
