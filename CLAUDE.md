# MEL Codebase Guidelines

## Build/Test Commands
- Run all tests: `pytest --doctest-modules`
- Run single test: `pytest tests/path/to/test_file.py::test_function_name`
- Run static analysis: `./meta/static_tests.sh`
- Run unit tests: `python3 -m nose mel/ --with-doc --doctest-tests --all-modules`
- Fix formatting: `./meta/autofix.sh`

## Code Style
- Formatting: Black with 79 character line limit
- Imports: Standard lib first, third-party second, project modules last (alphabetically sorted in groups)
- Naming: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- Types: Type hints for function parameters and return values where applicable
- Documentation: Google-style docstrings with Args/Returns sections for complex functions
- Error handling: Explicit exceptions with descriptive messages, exception chaining with `raise X from Y`
- Indentation: 4 spaces
- Linting: pylint, pyflakes, vulture

## Development Environment
- Python 3.8+
- Install dev dependencies: `pip install -e '.[dev]'`