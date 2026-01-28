# Local Code Agent - Architecture Documentation

## Overview

The Local Code Agent is a terminal-first, fully offline coding assistant that uses Ollama for LLM inference. It follows a **plan → execute → review → apply** workflow with mandatory human approval before making changes.

## Core Principles

1. **Offline First**: No cloud APIs, works entirely with local Ollama
2. **Diff-Only**: Never overwrites files directly, always generates unified diffs
3. **Human-in-the-Loop**: All changes require explicit user approval
4. **Safe by Default**: Validates, backs up, and can rollback changes
5. **Terminal-First**: Rich CLI interface, no web app or IDE extension

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CLI (agent/cli.py)                                  │   │
│  │  - Interactive prompts & confirmations               │   │
│  │  - Progress indicators                               │   │
│  │  - Rich terminal output                              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌─────────▼─────────┐
│   Planning      │    │    Execution      │
│   Layer         │    │    Layer          │
│                 │    │                   │
│ ┌────────────┐ │    │ ┌──────────────┐ │
│ │ Planner    │ │    │ │ Executor     │ │
│ │ (planner)  │ │    │ │ (executor)   │ │
│ └────────────┘ │    │ └──────────────┘ │
│                 │    │                   │
│ ┌────────────┐ │    │ ┌──────────────┐ │
│ │ Reviewer  │ │    │ │ Prompt Eng.  │ │
│ │ (reviewer)│ │    │ │ (prompt_eng.) │ │
│ └────────────┘ │    │ └──────────────┘ │
└────────┬───────┘    └─────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Core Infrastructure  │
         │                        │
         │ ┌────────────────────┐ │
         │ │ LLM Client         │ │
         │ │ (core/llm.py)      │ │
         │ │ - Retry logic      │ │
         │ │ - Error handling   │ │
         │ └────────────────────┘ │
         │                        │
         │ ┌────────────────────┐ │
         │ │ Config Management  │ │
         │ │ (core/config.py)   │ │
         │ └────────────────────┘ │
         │                        │
         │ ┌────────────────────┐ │
         │ │ Exceptions         │ │
         │ │ (core/exceptions)  │ │
         │ └────────────────────┘ │
         └───────────┬────────────┘
                     │
         ┌───────────▼───────────┐
         │   Tools & Utilities   │
         │                        │
         │ ┌────────────────────┐ │
         │ │ Patch Application  │ │
         │ │ (tools/apply_patch)│ │
         │ │ - Safe application │ │
         │ │ - Rollback support │ │
         │ └────────────────────┘ │
         │                        │
         │ ┌────────────────────┐ │
         │ │ Diff Generation    │ │
         │ │ (tools/diff)       │ │
         │ └────────────────────┘ │
         │                        │
         │ ┌────────────────────┐ │
         │ │ File Operations   │ │
         │ │ (tools/read_file)  │ │
         │ └────────────────────┘ │
         │                        │
         │ ┌────────────────────┐ │
         │ │ Git Integration   │ │
         │ │ (tools/git_*)      │ │
         │ └────────────────────┘ │
         └───────────┬────────────┘
                     │
         ┌───────────▼───────────┐
         │   Memory & Indexing   │
         │                        │
         │ ┌────────────────────┐ │
         │ │ Repository Scan    │ │
         │ │ (memory/index.py)  │ │
         │ │ - Git-aware        │ │
         │ │ - File discovery   │ │
         │ └────────────────────┘ │
         └────────────────────────┘
```

## Component Details

### 1. CLI Layer (`agent/cli.py`)

**Responsibilities**:
- Parse command-line arguments
- Orchestrate workflow (plan → execute → apply → review)
- Interactive user prompts and confirmations
- Progress indicators and rich output
- Error handling and user feedback

**Key Functions**:
- `run()`: Main entry point, orchestrates entire workflow
- `_confirm()`: User confirmation prompts
- `_run_one()`: Execute single task workflow

**Workflow**:
1. Validate inputs
2. Scan repository
3. Build plan
4. Show plan to user (if `--show-plan`)
5. Confirm execution
6. Generate diffs
7. Show diffs and get approval
8. Apply patches (with backups)
9. Run post-checks (optional)
10. Review results

### 2. Planning Layer (`agent/planner.py`)

**Responsibilities**:
- Analyze user task
- Select relevant files to read/modify
- Generate execution plan
- Infer file paths from task description

**Key Functions**:
- `build_plan()`: Main planning function
- `_score_file()`: Score file relevance based on keywords
- `_choose_files()`: Select top N relevant files
- `_infer_path_from_task()`: Extract file path from task

**Algorithm**:
1. Tokenize task into keywords
2. Score each file based on:
   - Keyword matches in filename
   - File type (.py, .md prioritized)
   - Path depth (shallow paths preferred)
3. Select top N files (default: 10)
4. Generate plan with steps and assumptions

### 3. Execution Layer (`agent/executor.py`)

**Responsibilities**:
- Generate code diffs from task + context
- Build context from selected files
- Validate generated diffs
- Support deterministic and LLM-based execution

**Key Functions**:
- `execute()`: Main execution function
- `build_context()`: Build context snippets from files
- `_build_prompt()`: Create LLM prompt
- `_extract_diffs()`: Extract diffs from LLM output
- `_validate_diffs()`: Validate diff format and safety

**Execution Strategies**:
1. **Deterministic**: Pattern matching for replace/append/insert
2. **LLM-Based**: Code generation from context + task

**Context Building**:
- Reads top N files (default: 5)
- Selects relevant snippets (max 4000 bytes)
- Prioritizes files with symbol matches
- Includes structural info (def/class lines)

### 4. Review Layer (`agent/reviewer.py`)

**Responsibilities**:
- Review generated diffs
- Identify risks and edge cases
- Suggest tests to run

**Key Functions**:
- `review_diffs()`: Generate review from diffs

**Current Implementation**: Basic (can be enhanced with LLM)

### 5. LLM Client (`core/llm.py`)

**Responsibilities**:
- Communicate with Ollama server
- Handle retries with exponential backoff
- Stream responses
- Error handling and recovery

**Key Functions**:
- `ask()`: Send prompt, get response (with retries)
- `ask_stream()`: Stream response chunks
- `_retry_with_backoff()`: Retry logic with exponential backoff

**Retry Strategy**:
- Max retries: 3 (configurable via `LCA_MAX_RETRIES`)
- Backoff: Exponential (base 1.5)
- Retryable errors: 429, 500, 502, 503, 504
- Non-retryable: Model not found (404 with model error)

### 6. Patch Application (`tools/apply_patch.py`)

**Responsibilities**:
- Parse unified diffs
- Apply hunks safely
- Create backups
- Rollback on failure

**Key Functions**:
- `apply_patch()`: Main patch application
- `_parse_diff()`: Parse diff into FilePatch objects
- `_apply_hunks()`: Apply hunks to file content
- `_is_binary()`: Detect binary files

**Safety Features**:
- Validates paths (within repo root)
- Rejects binary files
- Size limits (1MB per file, 2MB per patch)
- Creates `.bak` backups
- Atomic rollback on failure

### 7. Repository Scanning (`memory/index.py`)

**Responsibilities**:
- Discover files in repository
- Respect .gitignore
- Filter by extensions and size

**Key Functions**:
- `scan_repo()`: Main scanning function
- `_scan_with_git()`: Use git ls-files
- `_scan_with_walk()`: Fallback directory walk

**Strategy**:
1. Try `git ls-files` (respects .gitignore)
2. Fallback to `os.walk` with denylist
3. Filter by allowed extensions
4. Skip files >1MB

## Data Flow

### Typical Workflow

```
User Input
    ↓
CLI Validation
    ↓
Repository Scan → List[str] (file paths)
    ↓
Planning → Plan (files_to_read, files_to_modify, steps)
    ↓
User Confirmation
    ↓
Context Building → List[Dict] (snippets)
    ↓
Diff Generation → ExecutorResult (diffs, confidence)
    ↓
Diff Validation → Validated diffs
    ↓
User Approval
    ↓
Patch Application → List[str] (modified files)
    ↓
Post-Checks (optional)
    ↓
Review → Review (risks, suggestions)
```

### Diff Format

The agent uses **unified diff format**:

```
--- file.py
+++ file.py
@@ -start,count +start,count @@
 context line
-old line
+new line
 context line
```

## Error Handling

### Exception Hierarchy

```
LocalCodeAgentError (base)
├── LLMError (LLM communication issues)
├── PlanningError (planning phase errors)
├── ExecutionError (execution phase errors)
├── ConfigurationError (config issues)
└── SecurityError (security violations)
```

### Retry Strategy

- **LLM Calls**: Automatic retry with exponential backoff
- **Patch Application**: Rollback on failure, continue with other files
- **File Operations**: Graceful degradation, clear error messages

## Configuration

### Environment Variables

- `LCA_MODEL`: LLM model name (default: `qwen3-coder:latest`)
- `LCA_REPO_ROOT`: Repository root path (default: current directory)
- `LCA_MAX_PLAN_FILES`: Max files in plan (default: 50)
- `LCA_ALLOWED_EXTS`: Allowed file extensions (default: `.py,.md,.txt,.json,.yaml,.yml`)
- `LCA_DENYLIST_PATHS`: Denylisted paths (default: `.git,node_modules,.venv,...`)
- `LCA_MAX_RETRIES`: Max retry attempts (default: 3)
- `OLLAMA_HOST`: Ollama server URL (default: `http://127.0.0.1:11434`)

### Config Object

Centralized in `core/config.py`:
- Loads from environment variables
- Provides defaults
- Type-safe access

## Security Considerations

1. **Path Validation**: All paths validated against repo root
2. **Input Sanitization**: User inputs sanitized before use
3. **Size Limits**: Prevents patching large files
4. **Binary Detection**: Rejects binary files
5. **Backup & Rollback**: Safe change application

## Performance Considerations

1. **File Scanning**: Uses git when available (faster)
2. **Context Limits**: Max 5 files, 4000 bytes per snippet
3. **Caching**: (Future) Cache repository scans
4. **Parallelization**: (Future) Parallel file reading

## Extension Points

### Adding New Execution Modes

1. Add mode detection in `_infer_mode()`
2. Add mode-specific prompt in `_build_prompt()`
3. Update mode validation in CLI

### Adding New Tools

1. Create tool in `tools/` directory
2. Add to workflow in `agent/cli.py`
3. Add tests in `tests/`

### Adding New LLM Backends

1. Create backend class implementing `ask()` and `ask_stream()`
2. Update `core/llm.py` to use backend abstraction
3. Add config option

## Testing Strategy

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies (LLM, file system)
- Test edge cases and error conditions

### Integration Tests
- Test full workflow end-to-end
- Use mock LLM for deterministic results
- Test with real file system operations

### Coverage Goals
- Target: >80% code coverage
- Focus on critical paths (planning, execution, patching)

## Future Enhancements

1. **Semantic Code Search**: AST-based symbol extraction
2. **Learning System**: Task history and pattern recognition
3. **Multi-Step Tasks**: Task decomposition and dependency handling
4. **Better Context**: Smarter snippet selection
5. **Template System**: Common task templates

## Dependencies

### Core Dependencies
- `typer`: CLI framework
- `rich`: Terminal output formatting
- `httpx`: HTTP client for Ollama
- `GitPython`: Git operations

### Development Dependencies
- `pytest`: Testing framework
- `ruff`: Linting
- `mypy`: Type checking

## File Organization

```
local-code-agent/
├── agent/          # Agent logic (planning, execution, review)
├── core/           # Core infrastructure (LLM, config, exceptions)
├── memory/         # Repository understanding (scanning, indexing)
├── tools/          # Utilities (patching, diff, git)
├── tests/          # Test suite
└── main.py         # Entry point
```

## Design Decisions

1. **Diff-Only Approach**: Never overwrites files, always generates diffs for review
2. **Human Approval**: Mandatory confirmation before applying changes
3. **Offline First**: No cloud dependencies, works entirely locally
4. **Terminal-First**: Rich CLI instead of web app or IDE extension
5. **Safe by Default**: Multiple safety checks and rollback capabilities

---

*Last Updated: 2026-01-27*
