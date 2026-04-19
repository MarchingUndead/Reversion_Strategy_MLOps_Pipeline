# CLAUDE.md

Instructions for Claude Code working in this repository.

---

## Project at a glance

Quantitative trading strategy: detect statistical outliers in the futures-cash basis spread on Indian equity futures and predict whether they will revert, continue, or diverge. Pipeline produces a labeled events table from raw exchange tick data, which a model layer (deferred) consumes to generate live trade signals.

The strategy is in the middle of a clean-slate rewrite. The blueprint is `strategy_plan_v4.md`. Read it before doing substantive work.

---

## The plan is the spec

**`strategy_plan_v4.md` is the authoritative specification for this project.**

- DO read it before implementing anything non-trivial.
- DO refer back to it when in doubt about structure, contracts, or scope.
- DO NOT edit it. Ever. Not for typos, not for clarifications, not for "small" updates. If something in the plan seems wrong or out of date, surface the issue in chat — the human owns the plan.
- DO NOT edit this `CLAUDE.md` file without explicit instruction either. Same rule.

If a code change would require the plan to change to stay accurate, **stop and flag it** rather than proceeding. The plan drifting silently from the code is the failure mode to avoid.

---

## Critical invariants — these must hold at all times

These are not style preferences. Violating them silently breaks the strategy. Test for them where applicable.

### I1. No future data in distributions
For any session date `D`, the distribution snapshot used to score session `D` must contain **zero ticks from session `D` or later**. The streaming `DistStats` design enforces this by data flow: `update(D)` is called *after* `snapshot(as_of=D)` returns. Do not change that ordering. Do not add a "convenience" path that updates before scoring.

### I2. No future data in features
Features computed at detection index `i` must use only data at positions `≤ i` within the session, and only sessions strictly before the detection session for any cross-session lookup (e.g., prior-day VIX). The `features/snapshot.py` module is the only place feature computation lives. Adding a feature that violates causality is a strategy bug, not a code style issue.

### I3. Forward scan is preprocessing-only
Looking forward from a detection tick to find resolution is required for *labeling*. It happens only inside `events/scanner.py` during preprocessing. It must never happen during live signal generation. The `models/` interface receives features only — it cannot ask for future data.

### I4. The events schema is the contract
`events/schema.py` defines the columns of the labeled events parquet. Every consumer (label, train, backtest, live) reads from this schema. Adding a column means updating the schema module *first*, then the producer, then any consumer that needs the new column. Do not write columns directly into a DataFrame and hope downstream catches up.

### I5. The model interface is the contract
`models/base.py` defines `ReversionModel` (ABC) and `PredictDict` (output schema). Live and backtest consume any model through this interface. Do not bypass it. Do not let live or backtest depend on a specific model implementation. If you find yourself needing model-specific logic in the consumer, the interface is missing a method — extend the ABC, don't leak the abstraction.

### I6. Configuration is single-source
All tunable parameters live in `config.yaml` and are loaded through `core/config.py`. Do not hardcode thresholds, window sizes, file paths, or anything else that the plan documents as configurable. If you need a new tunable, add it to the config schema and the YAML defaults.

---

## Project structure (quick reference)

Per `strategy_plan_v4.md` §3:

```
src/strategy/
├── core/             utilities — config, logging, calendars (no business logic)
├── data/             disk I/O — readers, session loader, parquet helpers
├── distributions/    streaming dist stats — the leakage fix
├── features/         detection-time feature engineering (causal)
├── events/           outlier detection + labeling + schema
├── models/           CONTRACT ONLY at this stage — no implementations
├── pipelines/        orchestration — preprocess, label, live, backtest
└── cli.py            argparse subcommands
```

Dependency direction is one-way: `core → data → distributions/features → events → pipelines`. Nothing imports backwards. Tests (`tests/`) mirror this structure.

If the structure described in the plan does not match what's currently on disk, **the plan is correct and the code is wrong**. Update the code structure, not the plan.

---

## Conventions

### Logging
- Every module: `from strategy.core.logging import get_logger; log = get_logger(__name__)`
- No `print()` outside `cli.py`.
- INFO for pipeline progress, DEBUG for internal state, WARNING for recoverable issues, ERROR for failures.

### Config access
- `cfg = load_config()` once at the entry point, passed down explicitly.
- Do not call `load_config()` from inside library modules — they receive the relevant slice as a parameter.

### Type hints
- Public functions and class methods get type hints. Private helpers don't have to.
- `pd.DataFrame`, `pd.Timestamp`, `pd.Series` are fine (no need for stricter generics).

### Error handling
- Fail loud and early on contract violations (wrong columns, missing config keys, schema mismatches). Don't silently `try/except: pass`.
- Recoverable issues (missing optional file, no events for one symbol) get a WARNING and a continue.

### Naming
- Snake_case for everything Python.
- File names match module names match the dominant class/function inside.
- No abbreviations that aren't already in the plan (`dist`, `cfg`, `det_idx` are fine; new ones need justification).

---

## Workflow for making changes

1. **Identify which section of `strategy_plan_v4.md` covers the change.** If no section covers it, stop — the change is out of scope. Surface it in chat.
2. **Identify which package owns the change.** If multiple packages are affected, identify the order (lower in the dependency chain first).
3. **Update or add tests first** for any change touching distributions, events, or features. Other packages, tests are encouraged but not mandatory.
4. **Implement.**
5. **Run the affected tests, then the full suite.** All must pass.
6. **If the change introduced a new contract** (schema column, config key, interface method), confirm the plan already documents it. If not, stop and flag.

---

## Testing requirements

Per plan §8.

### Mandatory
The following tests must exist and pass before the corresponding pipeline is considered usable:

- `tests/distributions/test_no_leakage.py` — proves I1 holds.
- `tests/events/test_multi_day.py` — proves multi-day scanner handles overnight + expiry truncation correctly.
- `tests/events/test_labeler.py` — proves classification rules behave as specified.
- `tests/pipelines/test_preprocess_smoke.py` — end-to-end on synthetic fixtures.

### Encouraged
Anything else you write. Especially `core/`, `data/`, `features/`. Use `pytest` with fixtures from `tests/conftest.py`.

### Running tests
```
pytest                          # full suite
pytest tests/distributions      # one package
pytest -k no_leakage            # critical path only (fast feedback)
```

A change that breaks `test_no_leakage.py` is a strategy bug, not a test bug. Do not modify the test to make it pass — fix the code.

---

## Build order (from plan §9)

If the project is partially built, work in this order:

1. Skeleton (`pyproject.toml`, empty packages, working `python -m strategy --help`)
2. `core/`
3. `data/`
4. `distributions/` (with full test coverage including no-leakage tests)
5. `features/`
6. `events/`
7. `models/base.py` (interface only)
8. `pipelines/preprocess.py`
9. `pipelines/label.py`
10. `pipelines/live.py` (wired up to stub model)
11. `pipelines/backtest.py` (wired up to stub model)
12. `cli.py`
13. `scripts/verify_no_leakage.py`

Skipping ahead is allowed if a later step is needed to test an earlier one, but circular dependencies between packages are not.

---

## What Claude Code may NOT do without explicit instruction

- Edit `strategy_plan_v4.md`.
- Edit `CLAUDE.md` (this file).
- Edit anything under `project/` — that's the legacy code, frozen until cutover.
- Implement model classes. The `models/` package is interface-only at this stage. `RidgeReversionModel`, `EmpiricalKNNModel`, etc. are deferred to a separate work item.
- Add new dependencies beyond what's listed in plan §7 without flagging.
- Delete tests that are failing. Fix the code, or surface the test as outdated and ask.
- Restructure the package layout described in plan §3.
- Bypass the `DistStats` or `ReversionModel` interfaces with direct calls.
- Change the default distribution backend (currently `expanding`, per plan §11).

---

## What Claude Code SHOULD do proactively

- Read `strategy_plan_v4.md` at the start of any non-trivial task.
- Run the no-leakage tests after any change to `distributions/` or `pipelines/preprocess.py`.
- Flag drift between the code and the plan as soon as it's noticed.
- Suggest plan updates when proposing changes, but **wait for the human to update the plan** before implementing.
- Ask questions when an instruction is ambiguous, especially around the critical invariants.

---

## Open questions still on the table

Per plan §11, these have defaults but may change:

| # | Decision | Current default |
|---|---|---|
| 1 | Forward horizon (sessions) | 1.5 |
| 2 | Distribution backend default | expanding |
| 3 | `min_dwell_min` for labeling | 10 |
| 4 | Config validation approach | dataclass + manual |
| 5 | CLI library | argparse |
| 6 | Test framework | pytest |
| 7 | Per-symbol vs pooled | per-symbol |

If a task touches one of these, use the current default unless the human says otherwise. Do not unilaterally change a default.

---

## Quick command reference

```
pip install -e .                                # install package in dev mode
python -m strategy preprocess --symbol RELIANCE
python -m strategy label      --symbol RELIANCE
python -m strategy backtest   --symbol RELIANCE
python -m strategy live       --symbol RELIANCE
python -m strategy verify-leakage --symbol RELIANCE
pytest                                          # all tests
pytest -k no_leakage                            # critical path
```