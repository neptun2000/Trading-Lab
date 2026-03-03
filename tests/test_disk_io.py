import json
import pytest
import pandas as pd
from datetime import date
from pathlib import Path

from libb.core.reading_disk import DiskReader
from libb.core.writing_disk import DiskWriter
from libb.other.types_file import DiskLayout, Log


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def layout(tmp_path):
    root = tmp_path / "model"
    lay = DiskLayout.from_root(root)
    for d in [lay.portfolio_dir, lay.metrics_dir, lay.research_dir,
              lay.logging_dir, lay.deep_research_dir, lay.daily_reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return lay


@pytest.fixture
def reader(layout):
    return DiskReader(layout=layout)


@pytest.fixture
def writer(layout):
    return DiskWriter(layout=layout, run_date=date(2024, 1, 2))


def _init_full_disk(layout):
    """Write minimal valid files to every expected path."""
    layout.cash_path.write_text('{"cash": 5000.0}')
    layout.portfolio_path.write_text(
        "ticker,shares,buy_price,cost_basis,stop_loss,market_price,market_value,unrealized_pnl\n"
    )
    layout.portfolio_history_path.write_text("date,equity,cash,positions_value,daily_return_pct,overall_return_pct\n")
    layout.trade_log_path.write_text("date,ticker,action,shares,price,cost_basis,PnL,rationale,confidence,status,reason\n")
    layout.position_history_path.write_text("date,ticker,shares,avg_cost,stop_loss,market_price,market_value,unrealized_pnl\n")
    layout.pending_trades_path.write_text('{"orders": []}')
    layout.performance_path.write_text("[]")
    layout.behavior_path.write_text("[]")
    layout.sentiment_path.write_text("[]")


# ─────────────────────────────────────────────
# DiskReader
# ─────────────────────────────────────────────

class TestDiskReaderLoadCsv:
    def test_existing_file(self, layout, reader):
        layout.portfolio_path.write_text("ticker,shares\nAAPL,10\n")
        df = reader.load_csv(layout.portfolio_path)
        assert list(df.columns) == ["ticker", "shares"]
        assert len(df) == 1

    def test_missing_file_returns_empty(self, reader, layout):
        df = reader.load_csv(layout.portfolio_path)
        assert df.empty


class TestDiskReaderLoadJson:
    def test_existing_file(self, layout, reader):
        layout.sentiment_path.write_text('[{"score": 0.5}]')
        assert reader.load_json(layout.sentiment_path) == [{"score": 0.5}]

    def test_missing_file_returns_empty_list(self, reader, layout):
        assert reader.load_json(layout.sentiment_path) == []


class TestDiskReaderLoadOrdersDict:
    def test_existing_file(self, layout, reader):
        layout.pending_trades_path.write_text('{"orders": [{"ticker": "AAPL"}]}')
        assert reader.load_orders_dict(layout.pending_trades_path) == {"orders": [{"ticker": "AAPL"}]}

    def test_missing_file_returns_default(self, reader, layout):
        assert reader.load_orders_dict(layout.pending_trades_path) == {"orders": []}


class TestDiskReaderLoadCash:
    def test_valid_cash(self, layout, reader):
        layout.cash_path.write_text('{"cash": 12345.67}')
        assert reader.load_cash() == 12345.67

    def test_integer_cash(self, layout, reader):
        layout.cash_path.write_text('{"cash": 10000}')
        assert reader.load_cash() == 10000.0

    def test_missing_key_raises(self, layout, reader):
        layout.cash_path.write_text('{"balance": 1000}')
        with pytest.raises(RuntimeError, match="missing required key 'cash'"):
            reader.load_cash()

    def test_invalid_value_raises(self, layout, reader):
        layout.cash_path.write_text('{"cash": "not_a_number"}')
        with pytest.raises(RuntimeError, match="Invalid cash value"):
            reader.load_cash()


class TestDiskReaderSnapshot:
    def test_snapshot_captures_state(self, layout, reader):
        _init_full_disk(layout)
        snap = reader.save_disk_snapshot()
        assert snap.cash == 5000.0
        assert snap.pending_trades == {"orders": []}
        assert isinstance(snap.portfolio, pd.DataFrame)
        assert snap.sentiment == []

    def test_snapshot_deep_copies_dataframes(self, layout, reader):
        _init_full_disk(layout)
        snap = reader.save_disk_snapshot()
        # Mutate the snapshot's df — should not affect what was captured
        snap.portfolio["new_col"] = 1
        snap2 = reader.save_disk_snapshot()
        assert "new_col" not in snap2.portfolio.columns


# ─────────────────────────────────────────────
# DiskWriter
# ─────────────────────────────────────────────

class TestDiskWriterCash:
    def test_saves_and_reads_back(self, layout, writer, reader):
        _init_full_disk(layout)
        writer._save_cash(9999.99)
        assert reader.load_cash() == 9999.99


class TestDiskWriterOrders:
    def test_save_and_reload(self, layout, writer, reader):
        _init_full_disk(layout)
        orders = {"orders": [{"ticker": "AAPL", "action": "b"}]}
        writer.save_orders(orders)
        assert reader.load_orders_dict(layout.pending_trades_path) == orders


class TestDiskWriterResearch:
    def test_save_deep_research(self, layout, writer):
        path = writer.save_deep_research("Deep research content")
        assert path.exists()
        assert path.read_text() == "Deep research content"

    def test_save_daily_update(self, layout, writer):
        path = writer.save_daily_update("Daily update text")
        assert path.exists()
        assert path.read_text() == "Daily update text"

    def test_save_additional_log_write(self, layout, writer):
        writer.save_additional_log("notes.txt", "hello")
        p = layout.research_dir / "additional_logs" / "notes.txt"
        assert p.read_text() == "hello"

    def test_save_additional_log_append(self, layout, writer):
        writer.save_additional_log("notes.txt", "line1\n")
        writer.save_additional_log("notes.txt", "line2\n", append=True)
        p = layout.research_dir / "additional_logs" / "notes.txt"
        assert p.read_text() == "line1\nline2\n"


class TestDiskWriterLogging:
    def test_saves_json_log(self, layout, writer):
        log = Log(
            date="2024-01-02", weekday="Tuesday",
            started_at="2024-01-02T10:00:00", finished_at="2024-01-02T10:01:00",
            nyse_open_on_date=True, created_after_close=True,
            eligible_for_execution=True, processing_status="SUCCESS",
            orders_processed=2, orders_failed=0, orders_skipped=1,
            portfolio_value=10500.0, error="none",
        )
        writer._save_logging_file_to_disk(log)
        log_path = layout.logging_dir / "2024-01-02.json"
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert data["processing_status"] == "SUCCESS"
        assert data["orders_processed"] == 2


class TestDiskWriterSnapshot:
    def test_rollback_restores_csvs_and_json(self, layout, writer, reader):
        """_load_snapshot_to_disk restores CSVs and JSON files but not cash.json."""
        _init_full_disk(layout)
        writer.save_orders({"orders": [{"ticker": "AAPL"}]})
        snapshot = reader.save_disk_snapshot()

        # Mutate orders
        writer.save_orders({"orders": []})
        assert reader.load_orders_dict(layout.pending_trades_path) == {"orders": []}

        writer._load_snapshot_to_disk(snapshot)
        # Orders are restored
        assert reader.load_orders_dict(layout.pending_trades_path) == {"orders": [{"ticker": "AAPL"}]}

    def test_rollback_does_not_restore_cash(self, layout, writer, reader):
        """cash.json is not included in _load_snapshot_to_disk — documents known gap."""
        _init_full_disk(layout)
        snapshot = reader.save_disk_snapshot()
        assert snapshot.cash == 5000.0

        writer._save_cash(1.0)
        writer._load_snapshot_to_disk(snapshot)
        # cash.json is NOT restored by rollback
        assert reader.load_cash() == 1.0

    def test_rollback_restores_orders(self, layout, writer, reader):
        _init_full_disk(layout)
        original_orders = {"orders": [{"ticker": "AAPL"}]}
        writer.save_orders(original_orders)
        snapshot = reader.save_disk_snapshot()

        writer.save_orders({"orders": []})
        writer._load_snapshot_to_disk(snapshot)
        assert reader.load_orders_dict(layout.pending_trades_path) == original_orders
