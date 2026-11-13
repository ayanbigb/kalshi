"""How much risk can the singles strategy take and stay profitable? Answered walk-forward on real quotes.

Rules grid (gates + how hard to lean on the model):
  trust      multiplier on the blend's log-odds gap to the market (1 = the shipped 70/30 blend; >1 = lean further
             toward the model, which edge realization (lambda ~1.2) says is justified on average)
  min_edge   expected profit per contract after fees
  max_dis    largest |model - market| still treated as edge rather than missing news
  max_spread widest quote traded
Each week is traded with the rules that had the best log-growth on all EARLIER weeks (walk-forward), so the
reported result never uses the week it is scored on.

Sizing: fractional Kelly with per-trade and per-week caps on a running bankroll. Risk is measured by
resampling whole weeks (2,000 synthetic seasons): median and 5th-percentile ending bankroll, P(loss),
95th-percentile max drawdown. A sizing profile is called usable only if P(losing season) <= 20% and the
95th-percentile drawdown <= 40%.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from .config import REPORTS
from .edge import expit, fee_per_contract, logit

GRID = dict(trust=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5], min_edge=[0.015, 0.025, 0.04], max_dis=[0.15, 0.25], max_spread=[0.03, 0.06])


def _load() -> pd.DataFrame:
    d = pd.read_parquet(REPORTS / "market_backtest.parquet")
    d = d[d["p_market"].between(0.03, 0.97) & (d["games_career"] >= 6) & (d["inj_questionable"].fillna(0) < 1)].copy()
    d["spread"] = d["yes_ask"] - d["yes_bid"]
    d["lm"] = logit(d["p_market"].to_numpy())
    d["gap"] = logit(d["p_fixed"].to_numpy()) - d["lm"]
    d["t"] = d["season"] * 100 + d["week"]
    return d.reset_index(drop=True)


def bets(d: pd.DataFrame, trust: float, min_edge: float, max_dis: float, max_spread: float,
         min_price: float = 0.08, max_price: float = 0.92) -> pd.DataFrame:
    p = expit(d["lm"].to_numpy() + trust * d["gap"].to_numpy())
    ya, na = d["yes_ask"].to_numpy(), 1 - d["yes_bid"].to_numpy()
    cy, cn = ya + fee_per_contract(ya), na + fee_per_contract(na)
    evy, evn = p - cy, (1 - p) - cn
    yes = evy >= evn
    out = pd.DataFrame({"t": d["t"], "game_id": d["game_id"], "player_id": d["player_id"], "stat": d["stat"],
                        "side_yes": yes, "q": np.where(yes, p, 1 - p), "price": np.where(yes, ya, na),
                        "cost": np.where(yes, cy, cn), "ev": np.where(yes, evy, evn),
                        "win": np.where(yes, d["y"], 1 - d["y"])})
    ok = (out["ev"] >= min_edge) & (out["ev"] / out["cost"] >= 0.05) & out["price"].between(min_price, max_price) \
        & ((d["p_model"] - d["p_market"]).abs() <= max_dis).to_numpy() & (d["spread"] <= max_spread).to_numpy()
    b = out[ok].sort_values("ev", ascending=False).drop_duplicates(["t", "game_id", "player_id", "stat"])
    b["kelly"] = ((b["q"] - b["cost"]) / (1 - b["cost"])).clip(lower=0)
    b["r"] = b["win"] / b["cost"] - 1  # return per $1
    return b


def _week_growth(b: pd.DataFrame, frac: float = 0.25) -> pd.Series:
    """log growth per week with each bet sized frac*Kelly of a unit bankroll (weekly total capped at 50%)."""
    st = (frac * b["kelly"]).clip(upper=0.05)
    tot = st.groupby(b["t"]).transform("sum")
    st = st * np.minimum(1, 0.5 / tot.replace(0, np.nan)).fillna(1)
    return np.log1p((st * b["r"]).groupby(b["t"]).sum())


def walk_forward(min_weeks: int = 4, log=print) -> dict:
    d = _load()
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    cache = {i: bets(d, **c) for i, c in enumerate(combos)}
    growth = pd.DataFrame({i: _week_growth(b) for i, b in cache.items()}).fillna(0.0)
    weeks = sorted(d["t"].unique())
    chosen, picked = [], []
    for j, t in enumerate(weeks):
        if j < min_weeks:
            i = combos.index(dict(trust=1.0, min_edge=0.025, max_dis=0.25, max_spread=0.06))  # shipped defaults
        else:
            past = growth.loc[growth.index < t]
            # mean growth minus a penalty for its uncertainty, so a lucky rule needs more weeks to win
            score = past.mean() - 0.5 * past.std() / np.sqrt(len(past))
            i = int(score.idxmax())
        chosen.append(dict(t=int(t), **combos[i]))
        picked.append(cache[i][cache[i]["t"] == t])
    wf = pd.concat(picked, ignore_index=True)
    full = growth.mean().sort_values(ascending=False)
    best_all = combos[int(full.index[0])]
    log(f"  walk-forward: {len(wf)} bets over {wf['t'].nunique()} weeks; most-chosen rules: "
        f"{pd.DataFrame(chosen).drop(columns='t').value_counts().head(3).to_dict()}")
    return dict(wf_bets=wf, chosen=pd.DataFrame(chosen), best_in_sample=best_all, growth=growth, combos=combos, cache=cache)


def simulate(b: pd.DataFrame, frac: float, cap_trade: float, cap_week: float, bank0: float = 500.0,
             n_boot: int = 2000, seed: int = 0) -> dict:
    """Bankroll replay (in order) + week-resampled seasons. Stakes: frac*Kelly*bankroll, <= cap_trade*bankroll,
    weekly total <= cap_week*bankroll."""
    weeks = [g for _, g in b.groupby("t")]

    def run(order):
        bank, peak, mdd = bank0, bank0, 0.0
        for g in order:
            st = np.minimum(frac * g["kelly"].to_numpy(), cap_trade) * bank
            tot = st.sum()
            if tot > cap_week * bank:
                st *= cap_week * bank / tot
            bank += float((st * g["r"].to_numpy()).sum())
            peak = max(peak, bank); mdd = max(mdd, (peak - bank) / peak)
            if bank <= 1:
                break
        return bank, mdd

    actual, act_dd = run(weeks)
    rng = np.random.default_rng(seed)
    res = np.array([run([weeks[i] for i in rng.integers(0, len(weeks), len(weeks))]) for _ in range(n_boot)])
    end, dd = res[:, 0], res[:, 1]
    staked = float((np.minimum(frac * b["kelly"], cap_trade)).sum())
    return dict(frac=frac, cap_trade=cap_trade, cap_week=cap_week, bets=int(len(b)), weeks=len(weeks),
                actual_end=actual, actual_return=actual / bank0 - 1, actual_max_dd=act_dd,
                median_return=float(np.median(end) / bank0 - 1), p5_return=float(np.percentile(end, 5) / bank0 - 1),
                p_loss=float((end < bank0).mean()), p95_max_dd=float(np.percentile(dd, 95)),
                roi_flat=float(b["r"].mean()), staked_units=staked)


def risk_table(b: pd.DataFrame, bank0: float = 500.0) -> pd.DataFrame:
    rows = []
    for frac, cap_t, cap_w in [(0.10, 0.02, 0.3), (0.20, 0.05, 0.5), (0.35, 0.05, 0.6), (0.50, 0.08, 0.8),
                               (0.75, 0.10, 1.0), (1.00, 0.15, 1.0)]:
        rows.append(simulate(b, frac, cap_t, cap_w, bank0))
    t = pd.DataFrame(rows)
    t["usable"] = (t["p_loss"] <= 0.20) & (t["p95_max_dd"] <= 0.40)
    return t


def run(log=print) -> dict:
    wf = walk_forward(log=log)
    b = wf["wf_bets"]
    base = bets(_load(), 1.0, 0.025, 0.25, 0.06)
    out = dict(walk_forward=dict(bets=int(len(b)), roi=float(b["r"].mean()), win_rate=float(b["win"].mean())),
               shipped_rules=dict(bets=int(len(base)), roi=float(base["r"].mean())),
               best_in_sample_rules=wf["best_in_sample"],
               chosen_counts=wf["chosen"].drop(columns="t").value_counts().head(5).reset_index().to_dict("records"))
    rt = risk_table(b)
    out["risk_table"] = rt.to_dict("records")
    usable = rt[rt["usable"]]
    out["recommended"] = usable.sort_values("median_return", ascending=False).iloc[0].to_dict() if len(usable) else None
    (REPORTS / "risk_tuning.json").write_text(json.dumps(out, indent=1, default=float))
    wf["chosen"].to_csv(REPORTS / "risk_tuning_rules_by_week.csv", index=False)
    return dict(out, table=rt, wf=wf)
