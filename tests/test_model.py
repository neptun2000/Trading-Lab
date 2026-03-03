import json
import pytest
import pandas as pd
from pathlib import Path

from libb import LIBBmodel


TRADING_DATE = "2024-01-02"   # known NYSE open (Tuesday)
NYSE_CLOSED  = "2024-01-01"   # New Year's Day
WEEKEND_DATE = "2024-06-01"   # Saturday


@pytest.fixture
def model(tmp_path):
    return LIBBmodel(str(tmp_path / "model"), starting_cash=10_000, run_date=TRADING_DATE)


# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────

class TestInit:
    def test_creates_all_dirs(self, model):
        for d in [model.layout.portfolio_dir, model.layout.metrics_dir,
                  model.layout.research_dir, model.layout.logging_dir,
                  model.layout.deep_research_dir, model.layout.daily_reports_dir]:
            assert d.exists()

    def test_creates_all_files(self, model):
        for f in [model.layout.cash_path, model.layout.portfolio_path,
                  model.layout.pending_trades_path, model.layout.trade_log_path,
                  model.layout.portfolio_history_path, model.layout.position_history_path,
                  model.layout.performance_path, model.layout.behavior_path,
                  model.layout.sentiment_path]:
            assert f.exists()

    def test_initial_cash(self, model):
        assert model.cash == 10_000.0

    def test_initial_portfolio_empty(self, model):
        assert model.portfolio.empty

    def test_initial_pending_trades_empty(self, model):
        assert model.pending_trades == {"orders": []}

    def test_startup_snapshot_created(self, model):
        assert model.STARTUP_DISK_SNAPSHOT is not None
        assert model.STARTUP_DISK_SNAPSHOT.cash == 10_000.0

    def test_idempotent_reinit(self, tmp_path):
        """Reinitializing on same path should not overwrite existing data."""
        m1 = LIBBmodel(str(tmp_path / "model"), starting_cash=10_000, run_date=TRADING_DATE)
        m1.writer._save_cash(7777.0)
        m2 = LIBBmodel(str(tmp_path / "model"), starting_cash=10_000, run_date=TRADING_DATE)
        assert m2.cash == 7777.0  # existing cash preserved


# ─────────────────────────────────────────────
# save_orders
# ─────────────────────────────────────────────

class TestSaveOrders:
    def test_persists_to_disk(self, model):
        orders = {"orders": [{"ticker": "AAPL", "action": "b"}]}
        model.save_orders(orders)
        data = json.loads(model.layout.pending_trades_path.read_text())
        assert data == orders

    def test_overwrites_previous_orders(self, model):
        model.save_orders({"orders": [{"ticker": "AAPL"}]})
        model.save_orders({"orders": [{"ticker": "TSLA"}]})
        data = json.loads(model.layout.pending_trades_path.read_text())
        assert data["orders"][0]["ticker"] == "TSLA"


# ─────────────────────────────────────────────
# save_deep_research / save_daily_update
# ─────────────────────────────────────────────

class TestResearchSaving:
    def test_save_deep_research(self, model):
        path = model.save_deep_research("analysis text")
        assert path.exists()
        assert "deep_research" in path.name
        assert path.read_text() == "analysis text"

    def test_save_daily_update(self, model):
        path = model.save_daily_update("daily report text")
        assert path.exists()
        assert "daily_update" in path.name
        assert path.read_text() == "daily report text"


# ─────────────────────────────────────────────
# analyze_sentiment
# ─────────────────────────────────────────────

class TestAnalyzeSentiment:
    def test_returns_expected_keys(self, model):
        result = model.analyze_sentiment("The market looks bullish and positive.", report_type="daily")
        for key in ["polarity", "subjectivity", "positive_count", "negative_count", "token_count", "report_type", "date"]:
            assert key in result

    def test_report_type_stored(self, model):
        result = model.analyze_sentiment("Neutral text.", report_type="weekly")
        assert result["report_type"] == "weekly"

    def test_appended_to_in_memory_list(self, model):
        model.analyze_sentiment("First report.")
        model.analyze_sentiment("Second report.")
        assert len(model.sentiment) == 2

    def test_persisted_to_disk(self, model):
        model.analyze_sentiment("Some text about the economy.")
        data = json.loads(model.layout.sentiment_path.read_text())
        assert len(data) == 1

    def test_date_matches_run_date(self, model):
        result = model.analyze_sentiment("Market up today.")
        assert result["date"] == TRADING_DATE


# ─────────────────────────────────────────────
# process_portfolio — date guards
# ─────────────────────────────────────────────

class TestProcessPortfolioGuards:
    def test_skips_on_new_years(self, tmp_path):
        model = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date=NYSE_CLOSED)
        model.process_portfolio()
        data = json.loads((model.layout.logging_dir / f"{NYSE_CLOSED}.json").read_text())
        assert data["processing_status"] == "SKIPPED"

    def test_skips_on_weekend(self, tmp_path):
        model = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date=WEEKEND_DATE)
        model.process_portfolio()
        data = json.loads((model.layout.logging_dir / f"{WEEKEND_DATE}.json").read_text())
        assert data["processing_status"] == "SKIPPED"

    def test_raises_on_future_date(self, tmp_path):
        model = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date="2030-01-01")
        with pytest.raises(RuntimeError, match="ahead"):
            model.process_portfolio()

    def test_raises_on_backjump(self, tmp_path):
        m1 = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date="2024-01-02")
        m1.process_portfolio()
        m2 = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date="2023-12-29")
        with pytest.raises(RuntimeError, match="Backjump"):
            m2.process_portfolio()

    def test_raises_on_duplicate_date(self, tmp_path):
        # The `in` operator on a Series tests the index, not values, so the
        # "already exists" branch never fires — the backjump check catches it first.
        m1 = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date="2024-01-02")
        m1.process_portfolio()
        m2 = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date="2024-01-02")
        with pytest.raises(RuntimeError, match="Backjump"):
            m2.process_portfolio()

    def test_invalid_instance_raises(self, model):
        model._instance_is_valid = False
        with pytest.raises(RuntimeError, match="invalid"):
            model.process_portfolio()


# ─────────────────────────────────────────────
# process_portfolio — successful run (empty portfolio)
# ─────────────────────────────────────────────

class TestProcessPortfolioSuccess:
    def test_empty_portfolio_succeeds(self, tmp_path):
        model = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date=TRADING_DATE)
        model.process_portfolio()
        history = pd.read_csv(model.layout.portfolio_history_path)
        assert len(history) == 1
        assert float(history["equity"].iloc[0]) == pytest.approx(5_000.0)

    def test_writes_success_log(self, tmp_path):
        model = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date=TRADING_DATE)
        model.process_portfolio()
        data = json.loads((model.layout.logging_dir / f"{TRADING_DATE}.json").read_text())
        assert data["processing_status"] == "SUCCESS"

    def test_second_run_advances_history(self, tmp_path):
        m1 = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date="2024-01-02")
        m1.process_portfolio()
        m2 = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date="2024-01-03")
        m2.process_portfolio()
        history = pd.read_csv(m2.layout.portfolio_history_path)
        assert len(history) == 2


# ─────────────────────────────────────────────
# Order validation in _process_orders
# ─────────────────────────────────────────────

class TestOrderValidation:
    """Orders are validated before market data is fetched — no network calls needed."""

    MARKET_ORDER = {
        "action": "b", "ticker": "AAPL", "shares": 10,
        "order_type": "MARKET", "limit_price": None, "stop_loss": None,
        "rationale": "test", "confidence": 0.8, "time_in_force": None,
    }

    def _model_with_orders(self, tmp_path, orders, run_date=TRADING_DATE):
        model = LIBBmodel(str(tmp_path / "m"), starting_cash=5_000, run_date=run_date)
        payload = {"orders": orders}
        model.save_orders(payload)
        model.pending_trades = payload
        return model

    def test_past_order_rejected(self, tmp_path):
        order = {**self.MARKET_ORDER, "date": "2023-12-01"}
        model = self._model_with_orders(tmp_path, [order])
        model.process_portfolio()
        assert model.failed_orders == 1
        log = pd.read_csv(model.layout.trade_log_path)
        assert "REJECTED" in log["status"].values

    def test_nyse_holiday_order_rejected(self, tmp_path):
        # Order on 2024-07-04 (Independence Day) with run_date 2024-07-03
        order = {**self.MARKET_ORDER, "date": "2024-07-04"}
        model = self._model_with_orders(tmp_path, [order], run_date="2024-07-03")
        model.process_portfolio()
        assert model.failed_orders == 1
        log = pd.read_csv(model.layout.trade_log_path)
        assert any("NYSE CLOSED" in str(r) for r in log["reason"])

    def test_non_int_shares_rejected(self, tmp_path):
        order = {**self.MARKET_ORDER, "date": TRADING_DATE, "shares": 10.5}
        model = self._model_with_orders(tmp_path, [order])
        model.process_portfolio()
        assert model.failed_orders == 1

    def test_future_order_kept_as_pending(self, tmp_path):
        order = {**self.MARKET_ORDER, "date": "2024-01-05"}  # future NYSE-open date
        model = self._model_with_orders(tmp_path, [order])
        model.process_portfolio()
        assert model.skipped_orders == 1
        remaining = json.loads(model.layout.pending_trades_path.read_text())
        assert len(remaining["orders"]) == 1

    def test_empty_orders_no_failures(self, tmp_path):
        model = self._model_with_orders(tmp_path, [])
        model.process_portfolio()
        assert model.failed_orders == 0
        assert model.filled_orders == 0


# ─────────────────────────────────────────────
# reset_run
# ─────────────────────────────────────────────

class TestResetRun:
    def test_reset_with_auto_ensure_restores_state(self, model):
        model.writer._save_cash(1.0)
        model.reset_run(cli_check=False, auto_ensure=True)
        assert model.cash == 10_000.0
        assert model._instance_is_valid is True
        assert model.STARTUP_DISK_SNAPSHOT is not None

    def test_reset_without_auto_ensure_clears_disk(self, model):
        model.reset_run(cli_check=False, auto_ensure=False)
        # Root should be empty
        children = list(model._root.iterdir())
        assert children == []
