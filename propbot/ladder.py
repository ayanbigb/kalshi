"""How a Kalshi player-prop ladder and a multi-leg combo are priced, for one player or any combo you pick.

LADDER. Every rung "JSN 80+ receiving yards" is its own YES/NO contract paying $1. Its price is the market's
probability that the stat reaches that number, so the rungs of one player form a falling curve:
    P(80+) = 0.75, P(100+) = 0.45 ...  -> P(80 to 99 yards) = 0.75 - 0.45 = 0.30
A rung can never be priced above the one below it (100+ implies 80+). Buying YES at price c returns $1 when
it hits: profit (1 - c) per contract, payout multiple 1/c. Higher rungs: lower probability, bigger payout.
What makes a rung worth buying is not its probability but the gap between the true probability and the
price: EV per $1 = q / (c + fee) - 1. The best rung is the one with the largest EV (or Kelly growth), which
is often not the most likely one.

COMBO. A combo pays $1 only if every leg hits. With independent legs (different games) the probability is
the product. Inside one game it is not: QB passing yards and his WR's receiving yards rise together, two
receivers on one team compete for targets, a blowout lowers the winning QB's attempts. The model gives each
(player, stat) a correlated latent variable (copula fitted on out-of-sample residuals, combos.py) and
simulates the game 40,000 times; the combo's probability is the share of simulations where all legs hit.
All rungs of one player's ladder share one latent, so "JSN 80+" and "Lock 250+ pass yds" move together
the way the history says they do.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .edge import fee_per_contract

STAT_ALIASES = {
    "rec yds": "rec_yds", "receiving yards": "rec_yds", "rec_yds": "rec_yds", "recyds": "rec_yds",
    "rec": "rec", "receptions": "rec", "catches": "rec",
    "rush yds": "rush_yds", "rushing yards": "rush_yds", "rush_yds": "rush_yds",
    "rush att": "rush_att", "carries": "rush_att", "rush_att": "rush_att",
    "pass yds": "pass_yds", "passing yards": "pass_yds", "pass_yds": "pass_yds",
    "pass td": "pass_tds", "pass tds": "pass_tds", "passing td": "pass_tds", "passing tds": "pass_tds", "pass_tds": "pass_tds",
    "pass att": "pass_att", "attempts": "pass_att", "pass_att": "pass_att",
    "completions": "pass_comp", "pass comp": "pass_comp", "pass_comp": "pass_comp",
    "int": "pass_int", "ints": "pass_int", "interceptions": "pass_int", "pass_int": "pass_int",
    "rush+rec yds": "rr_yds", "rr yds": "rr_yds", "rr_yds": "rr_yds", "scrimmage yards": "rr_yds",
    "td": "td", "tds": "td", "touchdown": "td", "anytime td": "td",
}


def _mid(r):
    return (r["yes_bid"] + r["yes_ask"]) / 2


def ladder_view(px: pd.DataFrame, name: str, stat: str) -> pd.DataFrame:
    """One player's ladder: price, implied %, chance of landing between rungs, payout, model %, EV per rung."""
    s = px[px["name"].str.contains(name, case=False, na=False) & (px["stat"] == stat)].sort_values("strike")
    if s.empty:
        return s
    out = pd.DataFrame({
        "rung": (s["strike"] + 0.5).astype(int).astype(str) + "+",
        "yes_bid": s["yes_bid"].to_numpy(), "yes_ask": s["yes_ask"].to_numpy(),
        "market_%": (s["p_market"] * 100).to_numpy(),
        "model_%": (s["p_final"] * 100).to_numpy(),
    })
    nxt = np.r_[s["p_market"].to_numpy()[1:], 0.0]
    out["market_%_lands_here"] = (s["p_market"].to_numpy() - nxt) * 100  # P(between this rung and the next)
    ya, na = s["yes_ask"].to_numpy(), 1 - s["yes_bid"].to_numpy()
    cy, cn = ya + fee_per_contract(ya), na + fee_per_contract(na)
    q = s["p_final"].to_numpy()
    out["yes_payout_x"] = 1 / cy
    out["yes_profit_per_$10"] = 10 / cy - 10
    out["EV_yes_per_$"] = q / cy - 1
    out["EV_no_per_$"] = (1 - q) / cn - 1
    out["volume"] = s["volume"].to_numpy()
    return out.reset_index(drop=True)


def parse_leg(text: str) -> dict:
    """'Smith-Njigba rec yds 80+'  |  'Mahomes pass td 2+'  |  'Kelce rec 5+ no'  (no = the under)."""
    t = text.strip()
    side = "no" if re.search(r"\b(no|under)\b", t, re.I) else "yes"
    t = re.sub(r"\b(no|under|yes|over)\b", "", t, flags=re.I)
    m = re.search(r"(\d+)\s*\+?", t)
    if not m:
        raise ValueError(f"no number in leg '{text}'")
    n = int(m.group(1))
    rest = (t[:m.start()] + " " + t[m.end():]).strip()
    stat = None
    for alias in sorted(STAT_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", rest, re.I):
            stat = STAT_ALIASES[alias]
            rest = re.sub(r"\b" + re.escape(alias) + r"\b", "", rest, flags=re.I)
            break
    if stat is None:
        raise ValueError(f"no stat in leg '{text}' (try rec yds, rec, rush yds, pass yds, pass td, td, ...)")
    return dict(name=rest.strip(), stat=stat, strike=n - 0.5, side=side, text=text)


def find_legs(px: pd.DataFrame, specs: list[str]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        lg = parse_leg(spec)
        s = px[px["name"].str.contains(lg["name"], case=False, na=False) & (px["stat"] == lg["stat"])]
        hit = s[np.isclose(s["strike"], lg["strike"])]
        if hit.empty:
            avail = ", ".join(f"{int(k + 0.5)}+" for k in sorted(s["strike"].unique())) or "none"
            raise ValueError(f"'{spec}': no such rung on Kalshi right now (listed: {avail})")
        r = hit.sort_values("volume", ascending=False).iloc[0].to_dict()
        yes = lg["side"] == "yes"
        r.update(side=lg["side"], p_leg=r["p_final"] if yes else 1 - r["p_final"],
                 p_mkt=r["p_market"] if yes else 1 - r["p_market"],
                 price_leg=r["yes_ask"] if yes else 1 - r["yes_bid"],
                 label=f"{r['name']} {int(lg['strike'] + 0.5)}+ {lg['stat']}" + ("" if yes else " (NO)"))
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def price_combo(px: pd.DataFrame, specs: list[str], cfg: dict, n_draws: int = 40000) -> dict:
    """Price a combo you choose, leg by leg and jointly, with and without correlation."""
    from . import combos, strategy
    legs = find_legs(px, specs)
    cop = combos.load_copula()
    pr = combos.ComboPricer(legs, cop, n_draws)
    idx = tuple(range(len(legs)))
    p_model, p_fair = pr.price(idx)
    p_ind_model, p_ind_mkt = float(np.prod(legs["p_leg"])), float(np.prod(legs["p_mkt"]))
    ev = combos.evaluate(p_model, p_fair, legs, cfg)
    if len(legs) == 1:  # a single leg is just the contract itself: you pay its ask, no combo markup
        ev.update(est_price=float(legs["price_leg"].iloc[0]))
    risk = combos.risk_report(pr, idx, cfg)
    k = len(legs)
    cal = strategy.leg_factor(k)
    m = strategy.bet_math(p_model * cal, ev["est_price"])
    return dict(legs=legs, p_model=p_model, p_model_cal=p_model * cal, p_fair=p_fair, p_ind_model=p_ind_model,
                p_ind_mkt=p_ind_mkt, product_of_asks=float(np.prod(legs["price_leg"])), est_price=ev["est_price"],
                max_price=ev["max_price"], ev=m["ev_per_dollar"], kelly=m["kelly"], risk=risk, bet=m)
