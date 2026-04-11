# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

agent-claudia is a minimal Python web service (FastAPI + PostgreSQL) with a PostgreSQL configuration benchmark suite. Uses `uv` as the package manager with Python 3.12.

## Code Style

- Use type annotations on all Python code (function signatures, variables where not obvious).

## Common Commands

```bash
# Install dependencies
uv sync

# Run dev server (available at http://localhost:8000, docs at /docs)
uv run uvicorn main:app --reload

# Run PostgreSQL benchmark (requires PostgreSQL installed)
uv run postgres_bench/benchmark.py
uv run postgres_bench/benchmark.py --iterations 5 --rows 10000

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check
```

## Development Environment

A dev container is configured with Python 3.12 and PostgreSQL 16. Use it via VS Code or GitHub Codespaces for a ready-to-go setup.

## Architecture

- **main.py** — FastAPI app with two endpoints (`GET /` and `GET /health`)
- **postgres_bench/** — PostgreSQL configuration benchmarking tool:
  - `benchmark.py` — Manages the full benchmark lifecycle: creates template data directories per config (via `initdb`), then per-iteration copies the template, starts postgres, loads data with `asyncpg`, measures timing, and collects statistics (mean, median, stdev)
  - `configs/*.conf` — PostgreSQL configuration variants (baseline, async_commit, no_durability, minimal_memory, max_speed) with increasing performance/decreasing durability tradeoffs

## Dependencies

Key libraries: `fastapi`, `uvicorn`, `asyncpg`, `tabulate`. All managed via `uv` with versions pinned in `uv.lock`.
