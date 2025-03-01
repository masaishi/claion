# Build/Lint/Test Commands

## Environment Setup
- Setup virtual environment: `uv venv`
- Install dependencies: `uv pip install -e ".[dev]"`

## Lint & Type Checking
- Run Ruff linter: `uv run ruff check`
- Run Ruff formatter: `uv run ruff format`
- Run type checking: `uv run pyright`

## Testing
- Run all tests: `uv run pytest`
- Run single test: `uv run pytest tests/path/to/test_file.py::test_function_name`
- Run with coverage: `uv run pytest --cov=src`

# Code Style Guidelines

## Formatting & Imports
- Line length: 119 characters
- Use double quotes for strings
- Use space indentation
- Sort imports alphabetically (enforced by Ruff)

## Typing & Naming
- Use type hints for function parameters and return values
- Use snake_case for functions and variables
- Follow PEP8 conventions

## Error Handling
- Use explicit error handling with try/except
- Prefer specific exception types over generic exceptions

## Commits Style
- Please make your commits in detail.

### Branch Naming
- Use the following format for branch names: `type/description`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### Commit Messages
- Use the following format for commit messages: `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Scope: file or module name