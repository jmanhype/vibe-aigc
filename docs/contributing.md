# Contributing

Thank you for your interest in contributing to Vibe AIGC!

## Getting Started

### Fork and Clone

```bash
# Fork via GitHub UI, then:
git clone https://github.com/YOUR_USERNAME/vibe-aigc.git
cd vibe-aigc
```

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Verify setup
pytest -v
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code following the existing style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=vibe_aigc

# Run specific test file
pytest tests/test_planner.py -v
```

### 4. Check Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy vibe_aigc
```

### 5. Commit and Push

```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature-name
```

### 6. Open Pull Request

- Go to GitHub and create a PR
- Fill in the PR template
- Wait for CI checks to pass
- Address review feedback

## Commit Message Format

We use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Examples:
```
feat: add parallel execution for independent nodes
fix: handle empty workflow plans gracefully
docs: update quickstart guide with checkpoint example
```

## Code Style

- Use [Black](https://black.readthedocs.io/) for formatting
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Add type hints to all public functions
- Write docstrings for public APIs

```python
def my_function(param: str, count: int = 1) -> list[str]:
    """Short description of function.
    
    Longer description if needed.
    
    Args:
        param: Description of param
        count: Description of count
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
    """
    pass
```

## Testing Guidelines

- Write tests for all new functionality
- Use pytest fixtures for common setup
- Test both success and error cases
- Use async tests for async code:

```python
import pytest

@pytest.mark.asyncio
async def test_my_async_function():
    result = await my_async_function()
    assert result is not None
```

## Documentation

- Update relevant docs when changing functionality
- Add docstrings to new public APIs
- Include code examples where helpful
- Preview docs locally:

```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
