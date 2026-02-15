# Contributing to Enterprise Risk Intelligence Platform

Thank you for considering contributing to this project! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/RiskPrediction.git
cd RiskPrediction
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

## 🔄 Development Workflow

### 1. Make Changes

- Write clean, readable code
- Add type hints where appropriate
- Include docstrings for functions/classes
- Update tests for your changes

### 2. Run Quality Checks

```bash
# Format code
black src/ tests/ --line-length=100
isort src/ tests/ --profile=black

# Lint
flake8 src/ tests/

# Type check
mypy src/

# Security scan
bandit -r src/ -ll
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # or start htmlcov/index.html on Windows
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add feature description"
# or
git commit -m "fix: fix issue description"
```

#### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

## 🎨 Code Style

### Python Style Guide

- Follow PEP 8
- Use Black formatter (line length: 100)
- Use isort for import sorting
- Add type hints for function signatures
- Write Google-style docstrings

### Example Code

```python
from typing import List, Optional
import pandas as pd


def calculate_risk_score(
    financials: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Calculate risk scores for companies based on financial metrics.
    
    Args:
        financials: DataFrame with financial data
        weights: Optional dictionary of feature weights
        
    Returns:
        Series with risk scores for each company
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> df = pd.read_csv('financials.csv')
        >>> scores = calculate_risk_score(df)
    """
    if weights is None:
        weights = get_default_weights()
    
    # Implementation here
    return scores
```

## 🧪 Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings

### Test Example

```python
import pytest
import pandas as pd
from src.features import engineer_features


def test_engineer_features_creates_leverage_ratio():
    """Test that feature engineering creates leverage ratio."""
    # Arrange
    df = pd.DataFrame({
        'debt': [100, 200],
        'ebitda': [50, 100]
    })
    
    # Act
    result = engineer_features(df)
    
    # Assert
    assert 'leverage_ratio' in result.columns
    assert result['leverage_ratio'].iloc[0] == pytest.approx(2.0)
```

### Test Coverage

- Aim for 80%+ code coverage
- Critical functions: 100% coverage
- Run coverage reports locally before submitting

## 📝 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. Automated CI/CD checks run
2. Code review by maintainer
3. Address feedback if needed
4. Approval and merge

## 🐛 Issue Reporting

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Error messages/logs
- Screenshots if applicable

### Feature Requests

Include:
- Clear, descriptive title
- Problem it solves
- Proposed solution
- Alternatives considered
- Additional context

### Issue Template

```markdown
## Description
Clear description of the issue

## Environment
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- Package versions: [relevant packages]

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Screenshots/Logs
If applicable
```

## 📚 Additional Guidelines

### Documentation

- Update README.md for major changes
- Add docstrings to new functions/classes
- Update SETUP.md for setup changes
- Comment complex logic

### Performance

- Profile code for performance bottlenecks
- Use vectorized operations (pandas/numpy)
- Avoid unnecessary loops
- Cache expensive computations

### Security

- Never commit secrets/credentials
- Use environment variables
- Validate user inputs
- Run bandit security scanner

## 🙏 Thank You!

Your contributions make this project better! We appreciate your time and effort.

## 📞 Questions?

- Open a GitHub Discussion
- Comment on related issues
- Check existing documentation

---

**Happy Contributing! 🚀**
