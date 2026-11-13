"""Minimal Kalshi Trade API v2 client.

Market data is public (no key). Portfolio/order calls are signed with your API key id + private key
(RSA-PSS or Ed25519), read from KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH in .env.
"""
from __future__ import annotations

import base64
import time
import uuid

import requests

from .config import env

HOSTS = {"prod": "https://api.elections.kalshi.com", "demo": "https://demo-api.kalshi.co"}
PREFIX = "/trade-api/v2"


class KalshiError(RuntimeError):
    pass


class ReadOnlyError(KalshiError):
    pass


def orders_unlocked() -> bool:
    """Anything that is not a GET (orders, cancels, RFQs, quotes, transfers) is refused unless
    KALSHI_ALLOW_ORDERS=1 is set *in addition to* KALSHI_LIVE_TRADING=1. Default: read-only."""
    return env("KALSHI_ALLOW_ORDERS") == "1" and env("KALSHI_LIVE_TRADING") == "1"


class Kalshi:
    def __init__(self, environment: str = "prod"):
        self.env = environment
        self.host = HOSTS[environment]
        self.s = requests.Session()
        self.key_id = env("KALSHI_API_KEY_ID")
        self._key = None
        self._last = 0.0

    # ---- auth
    @property
    def has_key(self) -> bool:
        return bool(self.key_id and (env("KALSHI_PRIVATE_KEY_PATH") or env("KALSHI_PRIVATE_KEY")))

    @property
    def can_trade(self) -> bool:
        return self.has_key and orders_unlocked()

    def _private_key(self):
        """PEM from KALSHI_PRIVATE_KEY (the key text itself; literal \\n allowed, as environment settings
        store one line) or from the file at KALSHI_PRIVATE_KEY_PATH."""
        if self._key is None:
            from cryptography.hazmat.primitives import serialization
            text = env("KALSHI_PRIVATE_KEY")
            if text:
                data = text.replace("\\n", "\n").strip().encode()
            else:
                with open(env("KALSHI_PRIVATE_KEY_PATH"), "rb") as f:
                    data = f.read()
            self._key = serialization.load_pem_private_key(data, password=None)
        return self._key

    def _headers(self, method: str, path: str) -> dict:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ed25519, padding
        ts = str(int(time.time() * 1000))
        msg = (ts + method + path.split("?")[0]).encode()
        key = self._private_key()
        if isinstance(key, ed25519.Ed25519PrivateKey):
            sig = key.sign(msg)
        else:
            sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                           hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    # ---- transport
    def _req(self, method: str, path: str, params=None, body=None, auth=False):
        if method.upper() != "GET" and not orders_unlocked():
            raise ReadOnlyError(f"read-only: refused {method} {path} (set KALSHI_LIVE_TRADING=1 and KALSHI_ALLOW_ORDERS=1 to allow)")
        path = PREFIX + path
        for attempt in range(9):
            wait = 0.12 - (time.time() - self._last)  # stay well under the basic read limit
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.request(method, self.host + path, params=params, json=body, timeout=30,
                                   headers=self._headers(method, path) if auth else None)
            except requests.RequestException as e:
                if attempt == 8:
                    raise KalshiError(str(e))
                time.sleep(1 + attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60.0, 1.5 * 2 ** attempt))  # rate limited: back off hard, the archive pulls are long
                continue
            if r.status_code >= 400:
                raise KalshiError(f"{r.status_code} {method} {path}: {r.text[:300]}")
            return r.json() if r.text else {}
        raise KalshiError(f"gave up on {method} {path}")

    def _paged(self, path: str, key: str, params: dict, max_pages: int = 200) -> list[dict]:
        out, cursor = [], None
        for _ in range(max_pages):
            p = dict(params, **({"cursor": cursor} if cursor else {}))
            d = self._req("GET", path, params=p)
            out += d.get(key) or []
            cursor = d.get("cursor")
            if not cursor:
                break
        return out

    # ---- public market data
    def markets(self, series_ticker: str, status: str = "open", **extra) -> list[dict]:
        return self._paged("/markets", "markets", dict(series_ticker=series_ticker, status=status, limit=1000, **extra))

    def historical_markets(self, series_ticker: str) -> list[dict]:
        """Markets settled before Kalshi's archive cutoff (GET /historical/cutoff) live on a separate endpoint."""
        return self._paged("/historical/markets", "markets", dict(series_ticker=series_ticker, limit=1000), max_pages=2000)

    def market(self, ticker: str) -> dict:
        return self._req("GET", f"/markets/{ticker}")["market"]

    def orderbook(self, ticker: str) -> dict:
        return self._req("GET", f"/markets/{ticker}/orderbook")

    def fill_price(self, ticker: str, side: str, contracts: int) -> dict:
        """Walk the public order book: average price to BUY `contracts` of `side` right now.
        To buy YES you lift NO bids (yes price = 1 - no bid); to buy NO you lift YES bids."""
        ob = self.orderbook(ticker).get("orderbook_fp", {})
        other = ob.get("no_dollars" if side == "yes" else "yes_dollars") or []
        levels = sorted(((1 - float(p), float(q)) for p, q in other), key=lambda t: t[0])
        need, cost, filled = contracts, 0.0, 0
        for px, qty in levels:
            take = min(need, int(qty))
            cost += take * px; filled += take; need -= take
            if need <= 0:
                break
        return dict(filled=filled, avg_price=cost / filled if filled else float("nan"),
                    best=levels[0][0] if levels else float("nan"), worst=px if filled else float("nan"))

    def candles(self, tickers: list[str], start_ts: int, end_ts: int, interval_min: int = 60) -> dict[str, list]:
        out = {}
        for i in range(0, len(tickers), 100):
            d = self._req("GET", "/markets/candlesticks", params=dict(
                market_tickers=",".join(tickers[i:i + 100]), start_ts=start_ts, end_ts=end_ts,
                period_interval=interval_min, include_latest_before_start="true"))
            for m in d.get("markets", []):
                out[m.get("market_ticker")] = m.get("candlesticks", [])
        return out

    def historical_candles(self, ticker: str, start_ts: int, end_ts: int, interval_min: int = 60) -> list:
        """Candles of an archived market (one ticker per call; prices come without the `_dollars` suffix)."""
        return self._req("GET", f"/historical/markets/{ticker}/candlesticks",
                         params=dict(start_ts=start_ts, end_ts=end_ts, period_interval=interval_min)).get("candlesticks", [])

    # ---- authenticated
    def balance(self) -> float:
        d = self._req("GET", "/portfolio/balance", auth=True)
        return float(d["balance_dollars"]) if "balance_dollars" in d else d["balance"] / 100.0

    def positions(self) -> list[dict]:
        return self._req("GET", "/portfolio/positions", params={"count_filter": "position", "limit": 1000},
                         auth=True).get("market_positions", [])

    def _get_all(self, path: str, key: str, params: dict | None = None, max_pages: int = 50) -> list[dict]:
        out, cursor = [], None
        for _ in range(max_pages):
            p = dict(params or {}, limit=200, **({"cursor": cursor} if cursor else {}))
            d = self._req("GET", path, params=p, auth=True)
            out += d.get(key) or []
            cursor = d.get("cursor")
            if not cursor:
                break
        return out

    def account_snapshot(self) -> dict:
        """Everything readable about the account. GET requests only."""
        out = {}
        for name, fn in (
            ("balance", lambda: self._req("GET", "/portfolio/balance", auth=True)),
            ("positions", lambda: self._req("GET", "/portfolio/positions",
                                            params={"count_filter": "position", "limit": 1000}, auth=True)),
            ("orders", lambda: self._get_all("/portfolio/orders", "orders")),
            ("fills", lambda: self._get_all("/portfolio/fills", "fills", max_pages=10)),
            ("settlements", lambda: self._get_all("/portfolio/settlements", "settlements", max_pages=10)),
            ("rfqs", lambda: self._req("GET", "/communications/rfqs", auth=True)),
            ("quotes", lambda: self._req("GET", "/communications/quotes", auth=True)),
            ("api_limits", lambda: self._req("GET", "/account/limits", auth=True)),
        ):
            try:
                out[name] = fn()
            except KalshiError as e:
                out[name] = {"error": str(e)[:200]}
        return out

    def buy(self, ticker: str, side: str, count: int, price: float, ioc: bool = True) -> dict:
        """Buy `count` contracts of YES or NO at a limit `price` (dollars, e.g. 0.56).

        Kalshi's V2 order book is a single YES book: buying YES is a `bid` at the YES price,
        buying NO at q is an `ask` (selling YES) at 1 - q.
        """
        assert side in ("yes", "no") and 0.01 <= price <= 0.99 and count >= 1
        book_side, px = ("bid", price) if side == "yes" else ("ask", 1.0 - price)
        body = {"ticker": ticker, "client_order_id": str(uuid.uuid4()), "side": book_side,
                "count": f"{int(count)}.00", "price": f"{px:.4f}",
                "time_in_force": "immediate_or_cancel" if ioc else "good_till_canceled",
                "self_trade_prevention_type": "taker_at_cross"}
        return self._req("POST", "/portfolio/events/orders", body=body, auth=True)

    def cancel_all(self) -> dict:
        return self._req("DELETE", "/portfolio/events/orders", auth=True)
