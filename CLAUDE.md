# MEL Codebase Guidelines

## Build/Test Commands
- Run all tests: `pytest --doctest-modules`
- Run single test: `pytest tests/path/to/test_file.py::test_function_name`
- Run static analysis: `./meta/static_tests.sh`
- Run unit tests: `./meta/unit_tests.sh`
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

## Adding New Commands
- Commands are located in `mel/cmd/` directory
- Each command is a separate Python module with `setup_parser()` and `process_args()` functions
- Register new commands in `mel/cmd/mel.py` in the `COMMANDS` dictionary under appropriate category
- Import the command module at the top of `mel.py`
- Follow the pattern of existing commands like `rotomapagrelatelabel.py`
- Use `mel.rotomap.moles.load_image_moles()` to load mole data from images
- Use `mel.lib.image.load_image()` to load images
- Use `mel.rotomap.mask.load_or_none()` to load optional mask files

## MEL Project Structure
- `mel/cmd/` - CLI command implementations
- `mel/lib/` - Core utility libraries (image, math, UI, etc.)
- `mel/rotomap/` - Rotomap-specific functionality (moles, masks, detection, etc.)
- `mel/micro/` - Microscope image functionality
- `tests/` - Test files organized by module
- `meta/` - Development scripts (tests, linting, formatting)