# Contributing to SyntopicalChat

Thank you for considering contributing to SyntopicalChat! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- A detailed description of the bug
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Any relevant logs or screenshots
- Your environment (OS, Python version, etc.)

### Suggesting Features

If you have an idea for a new feature, please create an issue with the following information:

- A clear, descriptive title
- A detailed description of the feature
- Why this feature would be useful
- Any relevant examples or mockups

### Pull Requests

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/syntopicalchat.git
   cd syntopicalchat
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Code Style

We use the following tools to maintain code quality:

- Black for code formatting
- isort for import sorting
- mypy for type checking

Please ensure your code passes all checks before submitting a pull request:

```bash
black .
isort .
mypy .
```

## Testing

Please write tests for any new functionality. We use pytest for testing.

## Documentation

Please update the documentation to reflect any changes you make.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).