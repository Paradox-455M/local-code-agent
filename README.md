# Local Code Agent

Terminal-first, fully offline code agent that plans, proposes diffs, and applies them only after human approval. No web app, no cloud.

## Rules
- Works fully offline
- Uses Ollama as the only LLM runtime
- Must plan before editing
- Must output diffs only (no raw file overwrites)
- Human approval is mandatory before applying changes
- No web app, no VS Code extension, no cloud APIs
- Keep code minimal, readable, and testable

## Components
- `core/`: config + Ollama wrapper
- `memory/`: repo scanning
- `tools/`: safe read/diff/patch helpers + git views
- `agent/`: planner, executor (diff-only), reviewer
- `main.py`: CLI orchestrating plan → confirm → execute → confirm → apply → review

## Install & Run

### Method 1: Global Installation (Recommended via pipx)
This makes the `local-code-agent` command available everywhere in your terminal.
```bash
# Install pipx if you haven't already
brew install pipx
pipx ensurepath

# Install local-code-agent
pipx install .
```

### Method 2: Local Development (Virtual Environment)
Best for contributing to the project or keeping it isolated.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Usage
- Run the agent: `local-code-agent "describe your task"`  
- Focus on files: `local-code-agent "rename foo to bar" -f path/to/file.py`  
- Provide explicit instruction: `local-code-agent "task" -i "replace foo with bar"`  
- Review only: add `--dry-run` (no patches applied).
- Plan only: `--plan-only` to inspect plan/context without execution.
- Override model: `--model qwen3-coder:latest` (must be pulled in Ollama).
- Extra context: `-c path/to/file.py` or `--context-glob "src/**/*.py"`.
- Modes: `--mode bugfix|refactor|feature|general|auto` to tune prompts.
- Mock: `--mock-llm` for offline testing without calling the model.
- Post-check: `--post-check "python -m py_compile main.py"` runs after apply.
- Selective apply: `--selective-apply` to approve each diff before patching.
- Show plan/reasoning: defaults ON; use `--no-show-plan` / `--no-show-reasoning` to hide.
- Always plan first: `--always-plan` shows a brief plan summary and proceeds automatically (use `--show-plan` for full JSON). Apply confirmation still appears before patching.
- Preview context: defaults ON; use `--no-preview-context` to hide. `--symbols foo,bar` prioritizes symbols.
- Export patch: `--write-patch out.diff` to save diffs without applying.
- Mock replay: `--mock-from saved.jsonl` to replay stored LLM outputs.
- Relax validation: `--relaxed-validation` to loosen diff checks when LLM output is close.
- Allow outside repo: **OFF by default**; use `--allow-outside-repo` to permit patch application outside the repo root.

### Session Management (NEW!)
- **Start/continue a session**: `local-code-agent "task" --session mywork`
- **List all sessions**: `local-code-agent sessions list` or `local-code-agent sessions list --detailed`
- **View session history**: `local-code-agent sessions show <session_id>` or with `--full` for complete content
- **Delete a session**: `local-code-agent sessions delete <session_id>`
- **Export session to markdown**: `local-code-agent sessions export <session_id> -o output.md`
- Sessions persist conversation history, context files, and user preferences across multiple invocations

## Code-gen mode
- The executor can call the local LLM to synthesize unified diffs from task + context snippets.
- Guardrails: prefers not to touch tests unless requested; rejects malformed diffs; may ask for clarification if unsure.

## Safety and patching
- Patches use backups and rollback on failure; binary and size guards enabled.
- Optional hash guard via `apply_patch` for stricter safety (used internally as needed).
- If diffs look malformed, they are rejected; the agent may retry with a stricter prompt or ask for clarification.

## Performance & bottlenecks
- **Fast mode**: `LCA_FAST_MODE=1` skips knowledge graph, semantic search, and call-graph construction. Use for quicker runs when you don’t need deep codebase discovery.
- **Disable KG only**: `LCA_USE_KNOWLEDGE_GRAPH=0` turns off KG/semantic/call-graph but keeps other features. `LCA_FAST_MODE=1` implies this.
- **Explicit files**: `-f path/to/file.py` (or multiple `-f` paths) skips intelligent context building and uses basic context. Faster when you already know which files to touch.
- **Output token limit**: If diffs are truncated, set `LCA_NUM_PREDICT=8192` (or higher). Default `-1` means no limit.
- **Summary mode**: `LCA_SUMMARY_MODE=struct|llm|hybrid` controls per-file summaries. `struct` uses AST structure only; `llm` uses LLM map-reduce summaries; `hybrid` uses both.
- **Summary budgets**: `LCA_SUMMARY_MAX_BYTES`, `LCA_SUMMARY_CHUNK_LINES`, `LCA_SUMMARY_MAX_CHUNKS` tune summary size and map-reduce chunking.
- **Graph cache TTL**: `LCA_GRAPH_CACHE_TTL=300` controls how long KG/call-graph snapshots are reused.
- **Persistent caches**:
  - `.lca/summary.json` per-file summaries
  - `.lca/semantic_index/` embeddings + chunks
  - `.lca/graph_cache.json` KG/call-graph snapshot

## Troubleshooting
- Missing dependencies: install with `pip install -e .` (typer, rich, httpx, GitPython).
- Model not available: `ollama pull <model>` or set `LCA_MODEL`.
- Malformed diff / low confidence: provide clearer instructions, narrower file list (`-f`, `-c`, `--context-glob`), use `--symbols`, or relax validation (`--relaxed-validation`) and review carefully.
- Want to see reasoning: reasoning/plan/context are shown by default; to hide use `--no-show-reasoning`/`--no-show-plan`/`--no-preview-context`, or inspect `logs/run_*.jsonl`.
- Need to patch outside repo: default blocks it; add `--allow-outside-repo` to permit. Size/binary guards and apply confirmation still apply.

## Development
- Run tests:
  - `python -m venv .venv && . .venv/bin/activate`
  - `python -m pip install -e ".[dev]"`
  - `pytest -q`
- Lint: `ruff check .`
