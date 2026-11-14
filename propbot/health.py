"""Pre-trade health checks: refuse to trade when the inputs look wrong instead of betting on a data bug.

  model staleness   the model was last trained more than `max_model_staleness_weeks` weeks before this slate
  match rate        fewer than `min_match_rate` of open markets mapped to a player
  missing prices    more than 20% of matched markets got no model probability
  disagreement      the median |model - market| across the slate is above `max_median_disagreement`
                    (history: weekly medians 0.03-0.10, mean 0.052, 95th pct 0.071), the usual symptom of a
                    broken feed, a mis-joined player or a wrong starting QB
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import MODELS


def check(px: pd.DataFrame, cfg: dict, match_rate: float | None = None) -> list[str]:
    h = cfg.get("health", {})
    out = []
    try:
        meta = json.loads((MODELS / "meta.json").read_text())
        st = next(iter(meta["stats"].values()))
        trained = (int(st["last_season"]), int(st["last_week"]))
        from .data import current_season
        if trained[0] < current_season():
            out.append(f"model trained through {trained[0]} wk {trained[1]}, before this season")
        elif "week" in px.columns and len(px):
            lag = int(pd.to_numeric(px["week"], errors="coerce").max()) - trained[1]
            if lag > h.get("max_model_staleness_weeks", 3):
                out.append(f"model is {lag} weeks stale (trained through wk {trained[1]})")
    except Exception as e:
        out.append(f"model metadata unreadable ({type(e).__name__})")
    if match_rate is not None and match_rate < h.get("min_match_rate", 0.90):
        out.append(f"only {match_rate:.0%} of markets matched to players")
    if len(px):
        miss = px["p_model"].isna().mean()
        if miss > 0.20:
            out.append(f"{miss:.0%} of markets have no model probability")
        two = px.dropna(subset=["p_market", "p_model"])
        if len(two) >= 50:
            med = float((two["p_model"] - two["p_market"]).abs().median())
            if med > h.get("max_median_disagreement", 0.10):
                out.append(f"median |model - market| is {med:.3f} (history: 0.03-0.10)")
    return out


def role_change_flags(fut: pd.DataFrame, raw: dict, season: int) -> pd.DataFrame:
    """Per upcoming (game, player): 1 if this player's role is probably bigger than his history says, so the
    model's projection is stale. Two signals:
      mate_out   a same-team, same-position teammate who has been playing >= 35% of snaps is Out/Doubtful
                 (or Questionable without practicing all week), and this player has been playing < 60%
      depth_jump listed #1 on the depth chart but averaging < 40% of snaps
    The model systematically under-prices these players (seen 2026 wk 10: Bartholomew with Hockenson out,
    Kiner with Henderson out, Gould promoted), so they are skipped instead of bet against."""
    f = fut[["game_id", "team", "player_id", "position", "offense_pct_e6", "depth_rank", "week"]].copy()
    inj = raw["injuries"]
    weeks = f["week"].unique()
    wk = inj[(inj["season"] == season) & inj["week"].isin(weeks)]
    # Questionable after missing every practice usually means out (the report lags the news: 2026 wk 10
    # Kenneth Walker III was listed Questionable/DNP and ruled out the next day)
    dnp_q = wk["report_status"].eq("Questionable") & wk["practice_status"].astype(str).str.startswith("Did Not")
    out = wk[wk["report_status"].isin(["Out", "Doubtful"]) | dnp_q]
    ids = set(out["gsis_id"].dropna())
    share = fut.attrs.get("share_by_player")
    f["role_change"] = 0.0
    if ids and share is not None:
        o = out[["team", "gsis_id", "position"]].rename(columns={"gsis_id": "pid"})
        o["share"] = o["pid"].map(share)
        o = o[o["share"] >= 0.35]
        key = set(zip(o["team"], o["position"]))
        mate = np.array([(t, p) in key for t, p in zip(f["team"], f["position"])], dtype=bool)
        f.loc[mate & (f["offense_pct_e6"].fillna(0) < 0.60).to_numpy(), "role_change"] = 1.0
    f.loc[(f["depth_rank"] == 1) & (f["offense_pct_e6"].fillna(0) < 0.40) & f["position"].isin(["RB", "WR", "TE"]), "role_change"] = 1.0
    return f[["game_id", "player_id", "role_change"]]
