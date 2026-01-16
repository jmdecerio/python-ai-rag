# AGENTS.md

This repository is a small FastAPI-based RAG service. These notes help agentic coding tools
work consistently and safely in this codebase.

## Build / Lint / Test Commands

### Environment setup
- Create a virtualenv (optional): `python -m venv .venv`
- Activate it: `source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Required env var: `OPENAI_API_KEY=...`
- Optional env: `LM_STUDIO_API_BASE`, `LM_STUDIO_API_KEY` (see README)

### Run the app
- Dev server: `uvicorn app.main:app --reload`
- API docs: `http://127.0.0.1:8000/docs`

### Lint / format
- No linter or formatter is configured in the repo.
- If you add one, document the command here and keep it consistent.

### Tests
- No test framework is configured in the repo.
- If you add pytest:
  - Run all tests: `pytest`
  - Run a single file: `pytest tests/test_file.py`
  - Run a single test: `pytest tests/test_file.py::test_name`
  - Run by keyword: `pytest -k "keyword"`

## Code Style Guidelines

### Imports
- Keep standard library imports first, third-party next, local last.
- Prefer explicit imports over star imports.
- Use absolute imports within the `app/` package (e.g., `from .services import AIService`).
- Keep import blocks separated by a single blank line.

### Formatting
- Follow existing formatting style in files (4-space indent).
- Keep line length reasonable (target ~88–100 chars unless the file uses longer lines).
- Prefer multi-line argument lists for clarity when a call has many parameters.
- Avoid trailing whitespace and extra blank lines.

### Types
- Use type hints for public methods and return types.
- Use `List`, `Dict`, `Optional`, `Any` from `typing` for broad compatibility.
- Keep `np.ndarray` in type hints for embeddings.
- Ensure Pydantic models use explicit field types.

### Naming conventions
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions and methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Use descriptive names; avoid single-letter variables unless in short loops.

### Error handling
- Prefer raising `HTTPException` in FastAPI routes for API errors.
- Re-raise exceptions with context (`raise ... from exc`) when translating errors.
- Avoid swallowing exceptions; log or surface them appropriately.

### FastAPI patterns
- Keep routers in `controllers.py` and related services in `services.py`.
- Use Pydantic models for request/response payloads.
- Use `response_model` on routes for consistent API contracts.

### Data and storage
- CSV data lives in `data/` and is read in `RAGService`.
- Vector store is under `storage/` (ignored in git).
- Avoid changing CSV parsing unless necessary; keep encoding consistent.

### RAG behavior
- `RAGService` owns indexing and context formatting.
- `DatabaseService` handles ChromaDB interactions.
- `AIService` handles OpenAI embeddings and chat completion calls.
- Keep the embedding pipeline deterministic where possible.

### Configuration
- Settings live in `app/config.py` via `pydantic-settings`.
- Env vars are loaded from `.env` if present.
- Keep new configuration values centralized in `Settings`.

### Security and secrets
- Never commit API keys or `.env` files.
- Do not log or print `OPENAI_API_KEY`.
- Treat external API errors as user-facing failures in routes.

### Dependency management
- Add new dependencies to `requirements.txt` only.
- Prefer lightweight dependencies; keep the list minimal.

### Documentation
- Update `README.md` if you add new setup, commands, or major behaviors.
- Keep docs short and actionable.

### Git hygiene
- Keep diffs focused; avoid unrelated refactors.
- Do not commit `storage/`, `.venv/`, or `__pycache__/`.
- Prefer small, well-scoped changes.

### Miscellaneous
- Avoid adding inline comments unless they clarify non-obvious logic.
- Avoid adding new files unless requested or necessary.
- Maintain the project’s simple, direct coding style.

## Cursor / Copilot Rules
- No Cursor rules found (`.cursor/rules/` or `.cursorrules`).
- No Copilot instructions found (`.github/copilot-instructions.md`).
