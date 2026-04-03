# ratchet

Ratchet up static analysis strictness for this project by one notch.

## Process

1. Read pyproject.toml to understand currently enabled rules for both ruff and ty.

2. Pick ONE new rule category to enable. Choose from:
   - For ruff: a rule set from <https://docs.astral.sh/ruff/rules/> not yet in extend-select
   - For ty: a stricter setting not yet configured
   Prefer rules that are likely to have few existing violations (quick wins first).

3. Temporarily enable the rule and run the tool to see how many violations exist:
   - For ruff: `uv run ruff check` with the new rule
   - For ty: `ty check`

4. If violations are manageable (roughly <50), fix them all:
   - Use `uv run ruff check --fix` for auto-fixable ruff rules first
   - Manually fix remaining violations
   - Run `./meta/autofix.sh` to clean up formatting
   - Run `./meta/unit_tests.sh` to confirm nothing broke
   - Run `./meta/static_tests.sh` to confirm all checks pass

5. If violations are too numerous (>50), try a narrower sub-rule (e.g. "PLR1" instead of "PLR") or pick a different rule with fewer hits.

6. **Gradual approach** (for rules requiring large amounts of changes): If even
   sub-rules have too many violations, you can fix a batch of violations
   *without* enabling the rule in pyproject.toml. This lets us chip away at the
   debt incrementally across multiple invocations. Once the violation count
   reaches zero (or near-zero), enable the rule. Commit with message:
   "Fix N {RULE} violations: {description} (not yet enabled)"

7. Update pyproject.toml:
   - Add the new rule to extend-select (for ruff) or the config section (for ty)
   - Add a comment explaining what the rule category covers, matching existing comment style
   - Keep the list in the same order/style as existing entries

8. Commit with message: "Enable {RULE}: {description}"
   Example: "Enable DTZ: flake8-datetimez (enforce timezone-aware datetime usage)"

## Constraints

- Only enable ONE rule category per invocation
- Never disable or weaken existing rules
- All tests and static checks must pass before committing
- Follow the project's existing pyproject.toml comment style
- Update copyright year in changed files per CLAUDE.md
