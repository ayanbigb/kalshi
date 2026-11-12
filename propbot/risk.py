"""Risk engine: independent gates that must ALL pass before an order is sized.

  1. edge        - expected profit after fees clears min_edge and min_roi
  2. sanity      - price inside the tradeable band; model/market disagreement not so large that the
                   likeliest explanation is news the model has not seen
  3. liquidity   - two-sided quote, tight spread, enough size at the touch
  4. data        - player matched, enough history, not Questionable, game not about to start
  5. portfolio   - one rung per player ladder, per-player / per-game / total exposure caps
  6. breakers    - kill switch file, daily loss limit, max drawdown
"""
from __future__ import annotations

import math

import pandas as pd

from .config import ROOT
from .edge import fee

KILL_FILE = ROOT / "KILL"


def breakers(state: dict, r: dict) -> list[str]:
    out = []
    if KILL_FILE.exists():
        out.append("kill switch engaged")
    if state["pnl_today"] <= -r["daily_loss_limit"]:
        out.append(f"daily loss limit hit ({state['pnl_today']:.2f})")
    if state["drawdown"] >= r["max_drawdown"]:
        out.append(f"drawdown {state['drawdown']:.0%} >= {r['max_drawdown']:.0%}")
    return out


def gate_row(c: dict, r: dict) -> list[str]:
    """Per-candidate gates 1-4. Returns the list of failures (empty = pass)."""
    f = []
    if c["ev"] < r["min_edge"]:
        f.append("edge")
    if c["roi"] < r["min_roi"]:
        f.append("roi")
    if not (r["min_price"] <= c["price"] <= r["max_price"]):
        f.append("price_band")
    if abs(c["p_model"] - c["p_market"]) > r["max_disagreement"]:
        f.append("disagreement")
    if not (c["spread"] == c["spread"]) or c["spread"] > r["max_spread"]:
        f.append("spread")
    if not (c["size"] >= r["min_top_size"]):
        f.append("size")
    if c.get("games_career", 0) < r["min_games_history"]:
        f.append("history")
    if r["skip_questionable"] and c.get("inj_questionable", 0) >= 1:
        f.append("questionable")
    if c["minutes_to_kickoff"] < r["min_minutes_to_kickoff"]:
        f.append("kickoff")
    return f


def size_orders(cands: pd.DataFrame, open_trades: pd.DataFrame, state: dict, r: dict) -> pd.DataFrame:
    """Gate 5 + sizing. `cands` already passed gates 1-4. Greedy by expected value."""
    if cands.empty:
        return cands.assign(contracts=[], stake=[])
    bank = state["bankroll"]
    stake_of = lambda d: (d["contracts"] * d["price"] + d["fees"]) if len(d) else pd.Series(dtype=float)
    per_player = stake_of(open_trades).groupby(open_trades["player_id"]).sum().to_dict() if len(open_trades) else {}
    per_game = stake_of(open_trades).groupby(open_trades["game_id"]).sum().to_dict() if len(open_trades) else {}
    held = set(zip(open_trades["player_id"], open_trades["stat"])) if len(open_trades) else set()
    total, n_open = state["at_risk"], state["open"]
    out = []
    for c in cands.sort_values("ev", ascending=False).to_dict("records"):
        key = (c["player_id"], c["stat"])
        if key in held or n_open >= r["max_open_positions"]:
            continue  # rungs of one ladder are ~perfectly correlated: take only the best one
        room = min(r["max_stake_per_trade"],
                   r["max_stake_per_player"] - per_player.get(c["player_id"], 0.0),
                   r["max_stake_per_game"] - per_game.get(c["game_id"], 0.0),
                   r["max_total_exposure"] * bank - total,
                   state["cash"])
        stake = min(bank * r["kelly_fraction"] * c["kelly"], room)
        n = int(min(math.floor(stake / c["cost"]) if c["cost"] > 0 else 0, math.floor(c["size"])))
        if n < 1:
            continue
        fees = fee(c["price"], n)
        used = n * c["price"] + fees
        held.add(key); n_open += 1; total += used
        per_player[c["player_id"]] = per_player.get(c["player_id"], 0.0) + used
        per_game[c["game_id"]] = per_game.get(c["game_id"], 0.0) + used
        state = dict(state, cash=state["cash"] - used)
        out.append(dict(c, contracts=n, fees=fees, stake=used))
    return pd.DataFrame(out)
