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
- Install uv: `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Install dev dependencies: `uv venv && uv pip install -e '.[dev]'`

## Rotomap File System Structure
- **Mole files**: `{image}.jpg.json` - JSON arrays with mole coordinates and metadata
- **Mask files**: `{image}.jpg.mask.png` - Binary masks defining body regions
- **Meta files**: `{image}.jpg.meta.json` - Ellipse geometry data for coordinate transformation
- **Extra stem files**: `{image}.jpg.{stem}.json` - Additional mole data with specific stems

## Image Operations
- Use `mel.lib.image.save_image()` instead of `cv2.imwrite()` directly
- This function respects file permissions and prevents overwriting read-only files
- For JSON operations, use `mel.rotomap.moles.save_json()` for consistent formatting

## Command Structure
- Rotomap commands follow pattern: `mel rotomap {action} [options] FILES...`
- Add new commands to `mel/cmd/rotomap{action}.py` with `setup_parser()` and `process_args()` functions
- Register commands in `mel/cmd/mel.py` by importing module and adding to `COMMANDS["rotomap"]` dict

## Copyright and Generation
- In changed files, update copyright notice to include current year.
- Ensure "Generated with assistance from Claude Code." under copyright line.