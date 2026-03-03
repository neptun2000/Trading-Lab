# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LIBB (LLM Investor Behavior Benchmark) is a Python research library (`libb/`) for evaluating LLM-driven trading systems. It manages persistent portfolio state on disk and provides performance, sentiment, and behavioral metrics. The `user_side/` directory contains example user workflows that consume the library.

## Setup & Running

```bash
# Install (editable mode, inside venv)
pip install -r requirements.txt
pip install -e .

# Verify
python -c "import libb; print(libb.__file__)"

# Run example workflow
python -m user_side.workflow
```

Required environment variables: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`

There is no test suite yet — the project has limited automated test coverage by design.

## Architecture

### Entry Point

`LIBBmodel` (`libb/model.py`) is the single user-facing class, exported from `libb/__init__.py`. It wraps all portfolio state, disk I/O, metrics, and graphing.

```python
from libb import LIBBmodel
libb = LIBBmodel("path/to/run/model-name", starting_cash=10_000, run_date="2025-12-15")
```

### Execution Invariant

`libb.process_portfolio()` **must be called first** in every workflow. It:
1. Validates run date (no back-jumps, no future dates, no duplicate dates)
2. Checks NYSE market calendar
3. Processes pending orders via `libb/core/processing.py`
4. Downloads market data (yfinance/Stooq) and updates portfolio valuations
5. Appends to `portfolio_history.csv` and `position_history.csv`
6. On failure: rolls back disk state to the startup snapshot taken at `__init__`

### Core Layers

| Layer | Path | Responsibility |
|-------|------|----------------|
| Public API | `libb/model.py` | Orchestration, user-facing methods |
| Processing | `libb/core/processing.py` | Order processing, market data updates, history appending |
| Disk I/O | `libb/core/reading_disk.py`, `libb/core/writing_disk.py` | All file reads/writes |
| Execution | `libb/execution/` | Buy/sell/update-stop-loss logic, order validation |
| Metrics | `libb/metrics/` | Sentiment (`pysentiment2`), performance (Sharpe, Sortino, CAPM), behavior (WIP) |
| Graphs | `libb/graphs/` | Equity curves, sentiment overlays (matplotlib) |
| Types | `libb/other/types_file.py` | `Order`, `DiskLayout`, `ModelSnapshot`, `Log`, `TradeStatus` |
| Parsing | `libb/other/parse.py` | `parse_json(text, tag)` — extracts JSON from `<TAG>{...}</TAG>` blocks in LLM output |

### Disk Layout (`DiskLayout`)

Every model run gets an isolated directory tree auto-created by `LIBBmodel.__init__`:

```
<model_path>/
├── portfolio/
│   ├── cash.json                # authoritative cash balance
│   ├── pending_trades.json      # orders queued for next run
│   ├── portfolio.csv            # current positions
│   ├── portfolio_history.csv    # daily equity snapshots
│   ├── position_history.csv     # per-position daily history
│   └── trade_log.csv
├── metrics/
│   ├── performance.json
│   ├── sentiment.json
│   └── behavior.json
├── logging/                     # per-run JSON execution logs
└── research/
    ├── daily_reports/
    └── deep_research/
```

`DiskLayout.from_root(root)` in `libb/other/types_file.py` derives all paths from the root.

### Order Format

Orders are saved via `libb.save_orders(json_block)` where `json_block` is parsed from LLM output using `parse_json(text, "ORDERS_JSON")`. The `Order` TypedDict fields:

- `action`: `"b"` (buy), `"s"` (sell), `"u"` (update stop-loss)
- `order_type`: `"LIMIT"`, `"MARKET"`, `"UPDATE"`
- `ticker`, `shares` (must be `int`), `limit_price`, `stop_loss`, `date` (YYYY-MM-DD)
- `rationale`, `confidence` (0–1)

Orders dated before `run_date` or on NYSE holidays are REJECTED and logged.

### Rollback Safety

On `__init__`, a `ModelSnapshot` (frozen dataclass with deep copies of all DataFrames and dicts) is saved. If `process_portfolio()` raises, the snapshot is written back to disk, leaving state consistent.

## Key Design Principles

- **Explicit over implicit**: no hidden execution, no background threads, no magic
- **File-backed state**: all state lives on disk; in-memory mirrors it
- **User-controlled scheduling**: LIBB does not schedule runs — users call it
- **No heavy frameworks**: minimal external dependencies by design
- **Name mangling** (`_prefix`) discourages accidental mutation of internals but does not prevent access

## Constraints

- Market data source: yfinance only (Stooq partial support)
- No schema validation on LLM outputs — `parse_json` does basic extraction only
- Behavioral metrics not yet implemented (`behavior.json` is a placeholder)
- No automated tests; expect rough edges outside documented workflows
