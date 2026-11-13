"""Quote history: every scan's order-book top is appended to data/quotes/, so later scans can see how a
market has moved (parameters 212-218: price movement over 5 min / 1 h / 24 h / since first seen,
volatility, depth and a slippage estimate). Kalshi's API gives no free intraday history for props, so the
history starts when you start scanning; run `run --loop 15` or `scan` regularly to build it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA

QDIR = DATA / "quotes"
COLS = ["ticker", "ts", "yes_bid", "yes_ask", "yes_ask_size", "no_ask_size", "volume", "open_interest"]


def log_snapshot(mk: pd.DataFrame) -> None:
    if mk.empty:
        return
    QDIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    snap = mk[[c for c in COLS if c in mk.columns and c != "ts"]].assign(ts=now)
    snap.to_parquet(QDIR / f"{now:%Y%m%d_%H%M%S}.parquet")


def history(hours: float = 30) -> pd.DataFrame:
    if not QDIR.exists():
        return pd.DataFrame(columns=COLS)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    files = [f for f in sorted(QDIR.glob("*.parquet"))
             if pd.to_datetime(f.stem, format="%Y%m%d_%H%M%S").tz_localize("UTC") >= cutoff]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=COLS)


def add_movement(px: pd.DataFrame, stake_contracts: float = 50) -> pd.DataFrame:
    """Attach mid moves and volatility from the logged history, plus depth/slippage from the current book."""
    if px.empty:
        return px
    h = history()
    mid_now = (px["yes_bid"] + px["yes_ask"]) / 2
    if len(h):
        h = h.assign(mid=(h["yes_bid"] + h["yes_ask"]) / 2).sort_values("ts")
        now = pd.Timestamp.now(tz="UTC")
        def mid_at(delta):
            past = h[h["ts"] <= now - delta].groupby("ticker")["mid"].last()
            return px["ticker"].map(past)
        px["move_5m"] = mid_now - mid_at(pd.Timedelta(minutes=5))
        px["move_1h"] = mid_now - mid_at(pd.Timedelta(hours=1))
        px["move_24h"] = mid_now - mid_at(pd.Timedelta(hours=24))
        px["move_open"] = mid_now - px["ticker"].map(h.groupby("ticker")["mid"].first())
        px["mid_vol"] = px["ticker"].map(h.groupby("ticker")["mid"].std())
        px["quotes_seen"] = px["ticker"].map(h.groupby("ticker").size()).fillna(0)
    else:
        for c in ("move_5m", "move_1h", "move_24h", "move_open", "mid_vol"):
            px[c] = np.nan
        px["quotes_seen"] = 0.0
    size = np.where(px.get("side", pd.Series("yes", index=px.index)) == "no", px["no_ask_size"], px["yes_ask_size"])
    px["depth_top"] = size
    # paying the ask instead of the mid costs half the spread; size beyond the top level assumed 1c worse
    short = np.clip(stake_contracts - np.nan_to_num(size), 0, None) / stake_contracts
    px["slippage_est"] = (px["yes_ask"] - px["yes_bid"]) / 2 + 0.01 * short
    return px
