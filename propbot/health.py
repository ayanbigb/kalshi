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
