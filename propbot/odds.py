"""Optional sportsbook anchor via The Odds API (set ODDS_API_KEY in .env).

Sportsbook prop lines are the sharpest public signal there is. When available we don't replace the
model with them; we use the book's main line to *re-level* the model's ladder: price the book's own
line with the model, measure the log-odds gap to the de-vigged book probability, and apply (a fraction
of) that shift to every Kalshi strike for that player/stat.

NOTE: written against the documented v4 API but not exercised yet (no key was available at build time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .config import env
from .markets import norm_name

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKETS = {"player_pass_yds": "pass_yds", "player_pass_tds": "pass_tds", "player_pass_attempts": "pass_att",
           "player_pass_completions": "pass_comp", "player_pass_interceptions": "pass_int",
           "player_rush_yds": "rush_yds", "player_rush_attempts": "rush_att", "player_receptions": "rec",
           "player_reception_yds": "rec_yds", "player_rush_reception_yds": "rr_yds", "player_anytime_td": "td"}


def _prob(american: float) -> float:
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def fetch_props() -> pd.DataFrame:
    """Consensus de-vigged P(over line) per (player, stat). Empty frame if no key or on any failure."""
    key = env("ODDS_API_KEY")
    cols = ["nn", "stat", "line", "p_book", "books"]
    if not key:
        return pd.DataFrame(columns=cols)
    try:
        events = requests.get(f"{BASE}/events", params={"apiKey": key}, timeout=20).json()
        rows = []
        for ev in events:
            r = requests.get(f"{BASE}/events/{ev['id']}/odds", timeout=30, params={
                "apiKey": key, "regions": "us", "oddsFormat": "american", "markets": ",".join(MARKETS)})
            for bk in r.json().get("bookmakers", []):
                for m in bk.get("markets", []):
                    stat = MARKETS.get(m["key"])
                    by = {}
                    for o in m.get("outcomes", []):
                        by.setdefault((o.get("description"), o.get("point")), {})[o["name"]] = _prob(o["price"])
                    for (player, line), d in by.items():
                        if stat == "td" and "Yes" in d:      # one-sided market: strip a typical 2-way hold
                            rows.append((norm_name(player), stat, 0.5, d["Yes"] / 1.07, bk["key"]))
                        elif "Over" in d and "Under" in d and line is not None:
                            rows.append((norm_name(player), stat, float(line), d["Over"] / (d["Over"] + d["Under"]), bk["key"]))
        df = pd.DataFrame(rows, columns=["nn", "stat", "line", "p", "book"])
        if df.empty:
            return pd.DataFrame(columns=cols)
        main = df.groupby(["nn", "stat"])["line"].agg(lambda s: s.mode().iloc[0]).rename("main").reset_index()
        df = df.merge(main, on=["nn", "stat"])
        df = df[df["line"] == df["main"]]
        return df.groupby(["nn", "stat", "line"]).agg(p_book=("p", "median"), books=("book", "nunique")).reset_index()
    except Exception as e:  # never let an optional feed take the bot down
        print(f"  [odds] sportsbook feed unavailable: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=cols)


def logit_shift(p_model_at_line: float, p_book: float, weight: float) -> float:
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    return float(weight * (lg(p_book) - lg(p_model_at_line)))
