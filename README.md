# agent-claudia

A minimal Python web service built with [FastAPI](https://fastapi.tiangolo.com/) and PostgreSQL, managed by [uv](https://docs.astral.sh/uv/).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- PostgreSQL (or use the dev container)

## Development Setup

### Using the Dev Container (recommended)

Open this repository in VS Code and select **Reopen in Container**. The dev container includes:
- Python 3.12
- PostgreSQL 16 server binaries
- All project dependencies (installed via `uv sync`)

### Local Setup

```bash
# Install dependencies
uv sync

# Run the development server
uv run uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

Interactive API docs: http://localhost:8000/docs

## Project Structure

```
.
├── main.py            # FastAPI application entry point
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Locked dependency versions
└── .devcontainer/
    └── devcontainer.json  # Dev container configuration
```

## Dependencies

- **fastapi** – Web framework
- **uvicorn** – ASGI server
- **asyncpg** – Async PostgreSQL driver
