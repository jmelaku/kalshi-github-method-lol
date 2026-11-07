import base64
from datetime import datetime, timezone

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from btclab.book import parse_orderbook, quote_from_market
from btclab.kalshi import KalshiClient, parse_market, sign


def test_orderbook_bids_only_convention():
    q = parse_orderbook(
        {"orderbook_fp": {"yes_dollars": [["0.4000", "10.00"], ["0.4200", "5.00"]], "no_dollars": [["0.5500", "7.00"]]}}
    )
    assert q.yes_bid == 0.42  # best bid is the highest, regardless of input order
    assert q.yes_ask == 0.45  # 1 - best NO bid
    assert q.no_ask == 0.58  # 1 - best YES bid
    assert q.yes_ask_size == 7.0
    assert q.spread == 0.03


def test_orderbook_legacy_cents():
    q = parse_orderbook({"orderbook": {"yes": [[40, 10]], "no": [[55, 7]]}})
    assert q.yes_bid == 0.40 and q.yes_ask == 0.45


def test_empty_side():
    q = parse_orderbook({"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.9", "1"]]}})
    assert q.yes_bid is None and q.no_ask is None
    assert round(q.yes_ask, 4) == 0.1
    assert q.mid == 0.05  # one-sided: empty bid side treated as $0


def test_quote_from_market_fields():
    q = quote_from_market({"yes_bid_dollars": "0.3100", "yes_ask_dollars": "0.3400", "yes_ask_size_fp": "12.00"})
    assert q.yes_bid == 0.31 and q.yes_ask == 0.34 and q.yes_ask_size == 12.0


def test_parse_market_between_and_unsupported():
    m = parse_market(
        {
            "ticker": "KXBTC-26NOV0617-B80000",
            "event_ticker": "KXBTC-26NOV0617",
            "strike_type": "between",
            "floor_strike": 80000,
            "cap_strike": 80249.99,
            "close_time": "2026-11-06T22:00:00Z",
            "status": "active",
        }
    )
    assert m.contract.bounds() == (80000.0, 80249.99)
    assert m.seconds_to_close(datetime(2026, 11, 6, 21, 0, tzinfo=timezone.utc)) == 3600
    assert parse_market({"ticker": "X", "strike_type": "custom", "close_time": "2026-11-06T22:00:00Z"}) is None


def test_signature_verifies_and_ignores_query():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sig = sign(key, "1700000000000", "get", "/trade-api/v2/markets?limit=5")
    key.public_key().verify(
        base64.b64decode(sig),
        b"1700000000000GET/trade-api/v2/markets",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


class FakeResp:
    status_code = 201
    content = b"{}"
    text = "{}"

    def json(self):
        return {"order_id": "abc", "fill_count": "0.00"}


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url, json, headers))
        return FakeResp()


def test_buy_no_is_sent_as_yes_ask_and_signed_with_full_path():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    s = FakeSession()
    c = KalshiClient(key_id="kid", private_key=key, session=s, min_interval_s=0)
    c.create_order("KXBTC-X", "no", 3, 0.30)
    method, url, body, headers = s.calls[0]
    assert url.endswith("/trade-api/v2/portfolio/events/orders")
    assert body["side"] == "ask" and body["price"] == "0.7000" and body["count"] == "3"
    assert headers["KALSHI-ACCESS-KEY"] == "kid"
    assert c._path_prefix() == "/trade-api/v2"
