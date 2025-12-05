# Project Guidelines

## Package Manager

Use `uv` for Python package management:

- `uv add <package>` to add dependencies
- `uv add --dev <package>` to add dev dependencies
- `uv run <command>` to run commands in the virtual environment
- `uv sync` to sync dependencies

## Linting & Formatting

Use `ruff` for linting and formatting:

- `uv run ruff check .` to lint
- `uv run ruff check --fix .` to lint and auto-fix
- `uv run ruff format .` to format code
- `uv run ruff format --check .` to check formatting