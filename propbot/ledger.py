"""SQLite trade ledger shared by paper and live modes."""
from __future__ import annotations

import datetime as dt
import sqlite3

import pandas as pd

from .config import DATA

DB = DATA / "ledger.sqlite"
SCHEMA = """
CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mode TEXT, ticker TEXT, game_id TEXT, player_id TEXT,
  player TEXT, stat TEXT, strike REAL, side TEXT, contracts INTEGER, price REAL, fees REAL,
  p_model REAL, p_market REAL, p_final REAL, ev REAL, order_id TEXT,
  status TEXT DEFAULT 'open', payout REAL, pnl REAL, settled_ts TEXT);
"""


class Ledger:
    def __init__(self, path=DB):
        self.con = sqlite3.connect(path)
        self.con.executescript(SCHEMA)

    def add(self, **t) -> int:
        t.setdefault("ts", dt.datetime.now(dt.timezone.utc).isoformat())
        cols = ",".join(t)
        cur = self.con.execute(f"INSERT INTO trades({cols}) VALUES({','.join('?' * len(t))})", list(t.values()))
        self.con.commit()
        return cur.lastrowid

    def trades(self, mode: str, status: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM trades WHERE mode=?" + (" AND status=?" if status else "")
        return pd.read_sql(q, self.con, params=[mode] + ([status] if status else []))

    def settle(self, trade_id: int, yes_value: float):
        """yes_value = what one YES contract paid (1, 0, or a fair-price fraction if the player sat out)."""
        side, n, price, fees = self.con.execute(
            "SELECT side, contracts, price, fees FROM trades WHERE id=?", (trade_id,)).fetchone()
        payout = n * (yes_value if side == "yes" else 1 - yes_value)
        self.con.execute("UPDATE trades SET status='settled', payout=?, pnl=?, settled_ts=? WHERE id=?",
                         (payout, payout - n * price - fees, dt.datetime.now(dt.timezone.utc).isoformat(), trade_id))
        self.con.commit()

    def summary(self, mode: str, bankroll0: float) -> dict:
        t = self.trades(mode)
        s = t[t["status"] == "settled"].sort_values("settled_ts")
        o = t[t["status"] == "open"]
        realized = float(s["pnl"].sum()) if len(s) else 0.0
        at_risk = float((o["contracts"] * o["price"] + o["fees"]).sum()) if len(o) else 0.0
        eq = bankroll0 + s["pnl"].cumsum() if len(s) else pd.Series([bankroll0])
        hwm = max(bankroll0, float(eq.max()))
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        day = float(s.loc[s["settled_ts"].str[:10] == today, "pnl"].sum()) if len(s) else 0.0
        staked = float((s["contracts"] * s["price"] + s["fees"]).sum()) if len(s) else 0.0
        return dict(bankroll=bankroll0 + realized, cash=bankroll0 + realized - at_risk, at_risk=at_risk,
                    realized=realized, open=len(o), settled=len(s), wins=int((s["pnl"] > 0).sum()) if len(s) else 0,
                    roi=realized / staked if staked else 0.0, drawdown=(hwm - (bankroll0 + realized)) / hwm,
                    pnl_today=day)
