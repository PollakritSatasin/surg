# Contributing to SURG

Thank you for your interest in contributing to SURG! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/surg.git
   cd surg
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   pip install pytest black flake8
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

Format code with Black:
```bash
black surg/
```

Check with flake8:
```bash
flake8 surg/
```

## Testing

Run tests before submitting:
```bash
pytest tests/
```

Add tests for new features in the `tests/` directory.

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

5. Ensure all tests pass and code is properly formatted

## Adding New Features

When adding new features:

- Update documentation in README.md
- Add examples in the `examples/` directory
- Include unit tests
- Update configuration schema if needed

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- Operating system
- Minimal code to reproduce the issue
- Error messages and stack traces

## Questions?

Feel free to open an issue for questions or discussions.
