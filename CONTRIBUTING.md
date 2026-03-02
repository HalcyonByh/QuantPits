# Contributing to QuantPits

Thank you for your interest in contributing to QuantPits! Whether it's a bug fix, a new feature, or documentation improvement, we appreciate your effort to make this project better.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be considerate and constructive in all interactions.

## How Can I Contribute?

| Contribution Type | Description |
| --- | --- |
| 🐛 **Bug Fixes** | Fix issues reported in GitHub Issues |
| ✨ **New Features** | Add new models, scripts, or analytics capabilities |
| 📝 **Documentation** | Improve guides in `docs/`, README, or inline comments |
| 🧪 **Testing** | Add or improve tests for existing functionality |
| 🔧 **Refactoring** | Improve code quality without changing behavior |
| 🌐 **Translation** | Improve the Chinese (`README_zh.md`) or other translations |

## Getting Started

### Prerequisites

- Python 3.8 – 3.12
- A working [Microsoft Qlib](https://github.com/microsoft/qlib) installation
- (Optional) CUDA-compatible GPU + CuPy for brute force acceleration

### Setup

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/QuantPits.git
   cd QuantPits
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   conda create -n quantpits python=3.10
   conda activate quantpits
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

4. **Activate a workspace** to test your changes:
   ```bash
   source workspaces/Demo_Workspace/run_env.sh
   ```

## Development Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-description
   ```

2. **Make your changes** in small, focused commits.

3. **Test your changes** manually by running the relevant pipeline step in `Demo_Workspace`:
   ```bash
   # Example: test prediction pipeline
   python -m quantpits.scripts.prod_predict_only --all-enabled
   ```

4. **Push** to your fork and open a Pull Request.

## Coding Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use **4 spaces** for indentation (no tabs).
- Maximum line length: **120 characters**.
- Use meaningful, descriptive variable and function names.

### Project-Specific Conventions

- **Engine / Workspace separation**: All reusable logic belongs in `quantpits/`. Workspace-specific files (configs, data, outputs) must **never** be hardcoded — always resolve paths relative to the active workspace via `env.py`.
- **Configuration**: Use YAML files for workflow parameters. Avoid hardcoding magic numbers in scripts.
- **Logging**: Use Python's `logging` module; avoid bare `print()` statements in library code.
- **Imports**: Group imports in the standard order — stdlib, third-party, local — separated by blank lines.

### Documentation

- Update the relevant `docs/*.md` file if your change modifies user-facing behavior.
- All new scripts or significant functions should include docstrings.

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short description>

[optional body]
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

**Examples**:
```
feat(ensemble): add correlation-weighted fusion strategy
fix(order_gen): correct TopK selection when holdings are empty
docs(training): update incremental training guide with GPU notes
refactor(scripts): extract common arg parsing into shared module
```

## Pull Request Process

1. Ensure your branch is **up to date** with `main`:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Fill out the **Pull Request template** completely, including:
   - A clear description of *what* and *why*.
   - The type of change (bug fix / feature / docs / breaking change).
   - How you tested your changes.

3. **Review checklist** (from the PR template):
   - [ ] Code follows project style guidelines
   - [ ] Self-reviewed
   - [ ] Comments added where necessary
   - [ ] Documentation updated if applicable
   - [ ] No new warnings generated

4. A maintainer will review your PR. Please be responsive to feedback — we aim to review PRs within **one week**.

5. Once approved, a maintainer will **squash-merge** your PR into `main`.

## Reporting Bugs

Use the [Bug Report template](https://github.com/DarkLink/QuantPits/issues/new?template=bug_report.yml) to file a bug. Please include:

- Your Python version and OS
- Qlib version
- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs or error messages

## Suggesting Features

Open a GitHub Issue with the title prefixed by `[Feature Request]` and describe:

- **The problem** you're trying to solve.
- **Your proposed solution** and any alternatives you've considered.
- **Context** — which workspace setup or pipeline step is involved.

---

Thank you for contributing to QuantPits! 🎉

## Testing

This project uses `pytest` for unit testing. We encourage writing tests for all new features and bug fixes.

1.  **Framework:** `pytest` is the primary test runner. We also use `pytest-cov` for coverage reporting and `unittest.mock` for mocking dependencies.
2.  **Directory Structure:** Tests are located in the `tests/` directory, which mirrors the structure of the `quantpits/` package.
3.  **Running Tests:** Use the following command to run the test suite and generate a coverage report:
    ```bash
    pytest tests/ -v --cov=quantpits
    ```
4.  **Mocking:** When writing tests for modules that interact with the file system (`env.py`, `train_utils.py`) or external libraries like Qlib, use `unittest.mock.patch` or the `pytest-mock` plugin to isolate the unit under test. Global fixtures are defined in `tests/conftest.py`.

