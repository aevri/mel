# kaizen

Make one small, easily-reviewable improvement to the codebase.

## Goal

Identify and implement a single small improvement. The change should be
easy to review — aim for a diff that a reviewer can understand in under a
minute.

## Choosing an improvement

Pick ONE of these strategies. Prefer items near the top of the list.

1. **Do a small TODO.** Read `TODO.md` and grep the codebase for `TODO`,
   `FIXME`, `HACK`, `XXX` comments. Pick one that can be resolved with a
   small change (a few lines). Skip any that would require large refactors
   or design decisions.

2. **Add a missing test.** Look for modules or functions that have no
   corresponding tests, or find an edge case that existing tests don't
   cover. Write one focused test.

3. **Simplify a complicated function.** Find a function that is longer or
   more complex than it needs to be and simplify it slightly — extract a
   helper, remove dead branches, replace a manual loop with a builtin,
   etc. Do not rewrite whole modules.

4. **Improve type annotations.** Add type hints to a function that is
   missing them, or tighten overly broad types (e.g. `Any` → concrete
   type).

5. **Fix a code-quality nit.** Address a warning from ruff, ty, or
   vulture that is currently suppressed or missed. Or remove an unused
   import, variable, or dead code path that vulture reports.

## Constraints on change size

- Touch at most **3 files** (excluding test files and pyproject.toml).
- The diff should be **under ~60 changed lines** (excluding tests).
- If a TODO or improvement turns out to be bigger than expected, stop and
  pick a smaller one instead.
- When enabling a new check (linter rule, etc.), apply it to a **subset
  of files** if the full codebase has too many violations. For example,
  fix violations in one directory and add a per-file-ignores entry for
  the rest.

## Process

1. **Survey.** Spend a few minutes looking for candidate improvements:
   - `grep -r TODO mel/` and read `TODO.md`
   - Look at test coverage gaps
   - Run `ty check` and `uv run ruff check` for warnings
   - Scan for complex or poorly-typed functions
   Pick the single best candidate — the one with the highest
   value-to-size ratio.

2. **Implement.** Make the change. Keep it minimal and focused.

3. **Verify.**
   - Run `./meta/autofix.sh` to fix formatting.
   - Run `./meta/unit_tests.sh` to confirm tests pass.
   - Run `./meta/static_tests.sh` to confirm static checks pass.

4. **Commit.** Use a descriptive commit message in the style:
   `kaizen: {what you did}`
   Examples:
   - `kaizen: move new_image() from mel.lib.common to mel.lib.image`
   - `kaizen: add test for vec3 normalize edge case`
   - `kaizen: simplify moles.load_json error handling`

## Rules

- Only ONE improvement per invocation.
- Do not introduce new dependencies.
- Do not change user-facing CLI behavior unless fixing an obvious bug.
- All tests and static checks must pass before committing.
- Update copyright year in changed files per CLAUDE.md.
