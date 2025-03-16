# Contributing to Claion exp

Thank you for your interest in contributing to **Claion exp**!  
This guide will help you set up the development environment, ensure code consistency, run tests, and use the notebooks.


## 🚀 Quick Start

1. **Set up a virtual environment:**

We recommend using [`uv`](https://github.com/astral-sh/uv) for fast dependency management:

```bash
uv venv 
. .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**

For development:

```bash
uv pip install ".[dev]"
```

If you want to use Jupyter notebooks:

```bash
uv pip install ".[notebooks]"
```

3. **(Optional) Configure VSCode:**

Make sure to select the `.venv` virtual environment:
- Open the Command Palette (Ctrl+Shift+P)
- Search: `Python: Select Interpreter`
- Choose: `.venv`


## 📄 Formatting & Linting

We use **Ruff** for linting and formatting.

### Run Ruff (Lint & Auto-fix):

```bash
uv run ruff check --fix
```

## ✅ Type Checking

We use **Pyright** for type checking.

```bash
pyright
```


## 🧪 Testing

We use **pytest** for running tests with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```


## 📓 Notebooks Usage

To use or contribute to notebooks:

1. Install the extra notebook dependencies:

```bash
uv pip install ".[notebooks]"
```

2. Select the `.venv` kernel in VSCode notebook

## 📝 Pre-commit Hooks (Optional but Recommended)

Install pre-commit to automatically format and lint before commits:

```bash
uv run pre-commit install
```

Run manually:

```bash
uv run  pre-commit run --all-files
```


## 💡 Need Help?

Feel free to open an [Issue](https://github.com/masaishi/claion-exp/issues) or reach out!
