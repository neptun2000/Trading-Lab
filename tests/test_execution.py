import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch

from libb.execution.utils import is_nyse_open, append_log, catch_missing_order_data
from libb.execution.portfolio_editing import add_or_update_position, reduce_position, update_stoploss
from libb.execution.buy_logic import process_buy
from libb.execution.sell_logic import process_sell


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def trade_log(tmp_path):
    p = tmp_path / "trade_log.csv"
    p.write_text("date,ticker,action,shares,price,cost_basis,PnL,rationale,confidence,status,reason\n")
    return p


@pytest.fixture
def empty_portfolio():
    return pd.DataFrame(columns=[
        "ticker", "shares", "buy_price", "cost_basis",
        "stop_loss", "market_price", "market_value", "unrealized_pnl",
    ])


@pytest.fixture
def portfolio_with_aapl(empty_portfolio):
    return add_or_update_position(empty_portfolio, "AAPL", 10, 150.0, 130.0)


FAKE_MARKET = {
    "Ticker": "AAPL", "Low": 148.0, "High": 158.0,
    "Close": 155.0, "Open": 152.0, "Volume": 1_000_000,
}

BASE_ORDER = {
    "action": "b", "ticker": "AAPL", "shares": 10,
    "order_type": "LIMIT", "limit_price": 155.0, "stop_loss": 130.0,
    "date": "2024-01-02", "rationale": "test", "confidence": 0.8,
    "time_in_force": None,
}


# ─────────────────────────────────────────────
# is_nyse_open
# ─────────────────────────────────────────────

class TestNYSECalendar:
    def test_regular_trading_day(self):
        assert is_nyse_open(date(2024, 1, 2)) is True

    def test_christmas_closed(self):
        assert is_nyse_open(date(2024, 12, 25)) is False

    def test_new_years_closed(self):
        assert is_nyse_open(date(2024, 1, 1)) is False

    def test_saturday_closed(self):
        assert is_nyse_open(date(2024, 6, 1)) is False

    def test_sunday_closed(self):
        assert is_nyse_open(date(2024, 6, 2)) is False

    def test_accepts_timestamp(self):
        assert is_nyse_open(pd.Timestamp("2024-01-02").date()) is True


# ─────────────────────────────────────────────
# append_log
# ─────────────────────────────────────────────

class TestAppendLog:
    def test_append_dict(self, trade_log):
        append_log(trade_log, {"date": "2024-01-02", "ticker": "AAPL", "status": "FILLED"})
        df = pd.read_csv(trade_log)
        assert len(df) == 1
        assert df["ticker"].iloc[0] == "AAPL"

    def test_append_dataframe(self, trade_log):
        row_df = pd.DataFrame([{"date": "2024-01-02", "ticker": "TSLA", "status": "FILLED"}])
        append_log(trade_log, row_df)
        df = pd.read_csv(trade_log)
        assert df["ticker"].iloc[0] == "TSLA"

    def test_multiple_appends(self, trade_log):
        append_log(trade_log, {"ticker": "AAPL", "status": "FILLED"})
        append_log(trade_log, {"ticker": "TSLA", "status": "FAILED"})
        df = pd.read_csv(trade_log)
        assert len(df) == 2

    def test_missing_schema_raises(self, tmp_path):
        # File must not exist: load_df returns pd.DataFrame() → columns.empty=True → RuntimeError
        nonexistent_log = tmp_path / "no_header.csv"
        with pytest.raises(RuntimeError, match="Schema missing"):
            append_log(nonexistent_log, {"date": "2024-01-02"})

    def test_invalid_type_raises(self, trade_log):
        with pytest.raises(RuntimeError, match="Invalid data type"):
            append_log(trade_log, [1, 2, 3])


# ─────────────────────────────────────────────
# catch_missing_order_data
# ─────────────────────────────────────────────

class TestCatchMissingOrderData:
    def test_all_present_returns_true(self, trade_log):
        result = catch_missing_order_data(BASE_ORDER, ["ticker", "shares", "limit_price"], trade_log)
        assert result is True

    def test_missing_field_returns_false(self, trade_log):
        order = {**BASE_ORDER, "limit_price": None}
        assert catch_missing_order_data(order, ["limit_price"], trade_log) is False

    def test_missing_field_logs_failure(self, trade_log):
        order = {**BASE_ORDER, "shares": None}
        catch_missing_order_data(order, ["shares"], trade_log)
        df = pd.read_csv(trade_log)
        assert "MISSING" in df["reason"].iloc[0]


# ─────────────────────────────────────────────
# portfolio_editing
# ─────────────────────────────────────────────

class TestAddOrUpdatePosition:
    def test_add_new_position(self, empty_portfolio):
        df = add_or_update_position(empty_portfolio, "AAPL", 10, 150.0, 130.0)
        assert len(df) == 1
        assert df["ticker"].iloc[0] == "AAPL"
        assert df["shares"].iloc[0] == 10
        assert df["cost_basis"].iloc[0] == pytest.approx(1500.0)

    def test_add_to_existing_averages_cost(self, portfolio_with_aapl):
        df = add_or_update_position(portfolio_with_aapl, "AAPL", 5, 160.0, 130.0)
        assert len(df) == 1
        assert df["shares"].iloc[0] == 15
        assert df["cost_basis"].iloc[0] == pytest.approx(2300.0)  # 1500 + 800
        assert df["buy_price"].iloc[0] == pytest.approx(2300.0 / 15)

    def test_add_second_ticker(self, portfolio_with_aapl):
        df = add_or_update_position(portfolio_with_aapl, "TSLA", 5, 200.0, 180.0)
        assert len(df) == 2
        assert set(df["ticker"]) == {"AAPL", "TSLA"}


class TestReducePosition:
    def test_partial_reduce(self, portfolio_with_aapl):
        df, buy_price = reduce_position(portfolio_with_aapl, "AAPL", 4)
        assert df["shares"].iloc[0] == 6
        assert buy_price == pytest.approx(150.0)

    def test_full_reduce_removes_row(self, portfolio_with_aapl):
        df, _ = reduce_position(portfolio_with_aapl, "AAPL", 10)
        assert df.empty

    def test_returns_buy_price(self, portfolio_with_aapl):
        _, buy_price = reduce_position(portfolio_with_aapl, "AAPL", 3)
        assert buy_price == pytest.approx(150.0)


class TestUpdateStoploss:
    def test_updates_correctly(self, portfolio_with_aapl, trade_log):
        order = {**BASE_ORDER, "action": "u", "stop_loss": 145.0, "order_type": "UPDATE"}
        result = update_stoploss(portfolio_with_aapl, order, trade_log)
        assert result is True
        stop = portfolio_with_aapl.loc[portfolio_with_aapl["ticker"] == "AAPL", "stop_loss"].iloc[0]
        assert stop == pytest.approx(145.0)

    def test_missing_stop_loss_fails(self, portfolio_with_aapl, trade_log):
        order = {**BASE_ORDER, "action": "u", "stop_loss": None, "order_type": "UPDATE"}
        result = update_stoploss(portfolio_with_aapl, order, trade_log)
        assert result is False


# ─────────────────────────────────────────────
# buy_logic
# ─────────────────────────────────────────────

BUY_MARKET_PATH = "libb.execution.buy_logic.download_data_on_given_date"


class TestBuyLogic:
    def test_limit_buy_fills(self, empty_portfolio, trade_log):
        with patch(BUY_MARKET_PATH, return_value=FAKE_MARKET):
            df, cash, ok = process_buy(BASE_ORDER, empty_portfolio, 10_000.0, trade_log)
        assert ok is True
        assert len(df) == 1
        assert cash < 10_000.0

    def test_limit_buy_uses_open_when_below_limit(self, empty_portfolio, trade_log):
        # Open=152 < limit=155 → fill at open
        with patch(BUY_MARKET_PATH, return_value=FAKE_MARKET):
            df, cash, ok = process_buy(BASE_ORDER, empty_portfolio, 10_000.0, trade_log)
        assert cash == pytest.approx(10_000.0 - 10 * 152.0)

    def test_limit_buy_uses_limit_when_open_above(self, empty_portfolio, trade_log):
        # Open=160 > limit=155 → fill at limit
        market = {**FAKE_MARKET, "Open": 160.0}
        with patch(BUY_MARKET_PATH, return_value=market):
            df, cash, ok = process_buy(BASE_ORDER, empty_portfolio, 10_000.0, trade_log)
        assert cash == pytest.approx(10_000.0 - 10 * 155.0)

    def test_limit_buy_price_not_met(self, empty_portfolio, trade_log):
        order = {**BASE_ORDER, "limit_price": 140.0}  # Low=148 > 140 → won't fill
        with patch(BUY_MARKET_PATH, return_value=FAKE_MARKET):
            df, cash, ok = process_buy(order, empty_portfolio, 10_000.0, trade_log)
        assert ok is False
        assert df.empty

    def test_limit_buy_insufficient_cash(self, empty_portfolio, trade_log):
        with patch(BUY_MARKET_PATH, return_value=FAKE_MARKET):
            df, cash, ok = process_buy(BASE_ORDER, empty_portfolio, 100.0, trade_log)
        assert ok is False

    def test_market_buy_fills(self, empty_portfolio, trade_log):
        order = {**BASE_ORDER, "order_type": "MARKET", "limit_price": None}
        with patch(BUY_MARKET_PATH, return_value=FAKE_MARKET):
            df, cash, ok = process_buy(order, empty_portfolio, 10_000.0, trade_log)
        assert ok is True
        assert cash == pytest.approx(10_000.0 - 10 * FAKE_MARKET["Open"])

    def test_market_buy_insufficient_cash(self, empty_portfolio, trade_log):
        order = {**BASE_ORDER, "order_type": "MARKET", "limit_price": None}
        with patch(BUY_MARKET_PATH, return_value=FAKE_MARKET):
            _, _, ok = process_buy(order, empty_portfolio, 1.0, trade_log)
        assert ok is False

    def test_unknown_order_type_fails(self, empty_portfolio, trade_log):
        order = {**BASE_ORDER, "order_type": "FOO"}
        with patch(BUY_MARKET_PATH, return_value=FAKE_MARKET):
            _, _, ok = process_buy(order, empty_portfolio, 10_000.0, trade_log)
        assert ok is False

    def test_buy_logs_to_trade_log(self, empty_portfolio, trade_log):
        with patch(BUY_MARKET_PATH, return_value=FAKE_MARKET):
            process_buy(BASE_ORDER, empty_portfolio, 10_000.0, trade_log)
        df = pd.read_csv(trade_log)
        assert df["status"].iloc[0] == "FILLED"
        assert df["ticker"].iloc[0] == "AAPL"


# ─────────────────────────────────────────────
# sell_logic
# ─────────────────────────────────────────────

SELL_MARKET_PATH = "libb.execution.sell_logic.download_data_on_given_date"

SELL_ORDER = {
    "action": "s", "ticker": "AAPL", "shares": 5,
    "order_type": "LIMIT", "limit_price": 155.0, "stop_loss": None,
    "date": "2024-01-02", "rationale": "test", "confidence": 0.8,
    "time_in_force": None,
}


class TestSellLogic:
    def test_limit_sell_fills(self, portfolio_with_aapl, trade_log):
        with patch(SELL_MARKET_PATH, return_value=FAKE_MARKET):
            df, cash, ok = process_sell(SELL_ORDER, portfolio_with_aapl, 0.0, trade_log)
        assert ok is True
        assert df["shares"].iloc[0] == 5  # 10 - 5

    def test_limit_sell_uses_open_when_above_limit(self, portfolio_with_aapl, trade_log):
        # Open=152 < limit=155 → fill at limit
        market = {**FAKE_MARKET, "Open": 160.0}  # Open > limit → fill at open
        with patch(SELL_MARKET_PATH, return_value=market):
            df, cash, ok = process_sell(SELL_ORDER, portfolio_with_aapl, 0.0, trade_log)
        assert cash == pytest.approx(5 * 160.0)

    def test_limit_sell_price_not_met(self, portfolio_with_aapl, trade_log):
        order = {**SELL_ORDER, "limit_price": 200.0}  # High=158 < 200 → won't fill
        with patch(SELL_MARKET_PATH, return_value=FAKE_MARKET):
            _, _, ok = process_sell(order, portfolio_with_aapl, 0.0, trade_log)
        assert ok is False

    def test_insufficient_shares(self, portfolio_with_aapl, trade_log):
        order = {**SELL_ORDER, "shares": 100}  # only hold 10
        with patch(SELL_MARKET_PATH, return_value=FAKE_MARKET):
            _, _, ok = process_sell(order, portfolio_with_aapl, 0.0, trade_log)
        assert ok is False

    def test_market_sell_fills(self, portfolio_with_aapl, trade_log):
        order = {**SELL_ORDER, "order_type": "MARKET"}
        with patch(SELL_MARKET_PATH, return_value=FAKE_MARKET):
            df, cash, ok = process_sell(order, portfolio_with_aapl, 0.0, trade_log)
        assert ok is True
        assert cash == pytest.approx(5 * FAKE_MARKET["Open"])

    def test_full_sell_removes_position(self, portfolio_with_aapl, trade_log):
        order = {**SELL_ORDER, "shares": 10, "order_type": "MARKET"}
        with patch(SELL_MARKET_PATH, return_value=FAKE_MARKET):
            df, _, ok = process_sell(order, portfolio_with_aapl, 0.0, trade_log)
        assert ok is True
        assert df.empty

    def test_unknown_order_type_fails(self, portfolio_with_aapl, trade_log):
        order = {**SELL_ORDER, "order_type": "WEIRD"}
        with patch(SELL_MARKET_PATH, return_value=FAKE_MARKET):
            _, _, ok = process_sell(order, portfolio_with_aapl, 0.0, trade_log)
        assert ok is False

    def test_sell_logs_pnl(self, portfolio_with_aapl, trade_log):
        with patch(SELL_MARKET_PATH, return_value=FAKE_MARKET):
            process_sell(SELL_ORDER, portfolio_with_aapl, 0.0, trade_log)
        df = pd.read_csv(trade_log)
        assert df["status"].iloc[0] == "FILLED"
        # PnL = 5 * fill_price - 5 * 150 (buy_price)
        assert "PnL" in df.columns
