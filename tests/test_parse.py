import pytest
from libb.other.parse import parse_json


class TestParseJson:
    def test_basic_block(self):
        text = '<ORDERS_JSON>{"action": "b", "ticker": "AAPL"}</ORDERS_JSON>'
        assert parse_json(text, "ORDERS_JSON") == {"action": "b", "ticker": "AAPL"}

    def test_with_surrounding_text(self):
        text = 'Blah blah <ORDERS_JSON>{"x": 1}</ORDERS_JSON> more text'
        assert parse_json(text, "ORDERS_JSON") == {"x": 1}

    def test_multiline_json(self):
        text = '<ORDERS_JSON>{\n  "action": "b",\n  "ticker": "AAPL"\n}</ORDERS_JSON>'
        assert parse_json(text, "ORDERS_JSON") == {"action": "b", "ticker": "AAPL"}

    def test_trailing_comma_stripped(self):
        text = '<ORDERS_JSON>{"action": "b", "ticker": "AAPL",}</ORDERS_JSON>'
        assert parse_json(text, "ORDERS_JSON") == {"action": "b", "ticker": "AAPL"}

    def test_missing_tag_raises(self):
        with pytest.raises(ValueError, match="No ORDERS_JSON block found"):
            parse_json("no tag here", "ORDERS_JSON")

    def test_different_tag(self):
        text = '<REPORT_JSON>{"score": 0.9}</REPORT_JSON>'
        assert parse_json(text, "REPORT_JSON") == {"score": 0.9}

    def test_nested_json(self):
        text = '<ORDERS_JSON>{"data": {"key": "val"}}</ORDERS_JSON>'
        assert parse_json(text, "ORDERS_JSON") == {"data": {"key": "val"}}

    def test_whitespace_around_json(self):
        text = '<ORDERS_JSON>  {"action": "s"}  </ORDERS_JSON>'
        assert parse_json(text, "ORDERS_JSON") == {"action": "s"}

    def test_numeric_and_bool_values(self):
        text = '<ORDERS_JSON>{"shares": 10, "confidence": 0.8, "active": true}</ORDERS_JSON>'
        result = parse_json(text, "ORDERS_JSON")
        assert result["shares"] == 10
        assert result["confidence"] == 0.8
        assert result["active"] is True
