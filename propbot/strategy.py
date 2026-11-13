"""The math behind what the bot does with combos.

1. Edge realization (how much of the model's disagreement with Kalshi actually shows up in outcomes)
   On settled markets, fit  P(yes) = expit( logit(p_market) + lambda * (logit(p_blend) - logit(p_market)) ).
   lambda = 1: the blend's edge is real; 0: the market was right; >1: the blend is still too timid.
   Pooled lambda with a game-bootstrap interval, plus per-stat lambdas shrunk toward the pooled one.
   Combo legs use  p_robust = expit(logit(p_mkt) + lambda_lo * gap), lambda_lo = the 10th percentile, so
   the stake is sized on an edge the data supports with ~90% confidence, not on the point estimate.

2. Combo backtest on real prices, by leg count
   Every week of 2025-26 with settled Kalshi quotes: build candidate legs from the walk-forward model and
   the real pre-kickoff mids, run the same optimizer the bot runs (correlations fitted only on earlier
   seasons, lambda only on earlier weeks), keep the best few combos of each size 2..6, and settle them on
   what actually happened. Kalshi does not publish combo quotes, so each combo is priced at its
   correlation-aware fair value x (1 + hold) and results are shown for several holds.

3. Staking plan
   Per leg count: payout multiple, EV and standard deviation per $1, full Kelly fraction, the stake at
   1/4 and 1/2 Kelly, the stake needed for a target return, and a Monte-Carlo season (with the hit rate
   itself uncertain: drawn from a Beta posterior of the backtest) for P(profit), P(hitting the target),
   drawdowns and risk of ruin.

4. Policy (what `combo-run` does)
   Take the optimizer's shortlist, re-price every leg with p_robust, keep combos whose robust EV at the
   expected quote clears min_roi and whose leg count is allowed, size at kelly_fraction x robust Kelly with
   per-combo and weekly caps, and record them as PAPER combos (combo orders go through Kalshi's RFQ flow,
   which this code does not automate). `combo-settle` settles them leg by leg.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .config import DATA, MODELS, REPORTS, load_config
from .edge import expit, fee_per_contract, logit

ER_PATH = MODELS / "edge_realization.json"


# ----------------------------------------------------------------------------- 1. edge realization
def _fit_lambda(y, off, x) -> float:
    def nll(a):
        p = np.clip(expit(off + a * x), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(-2, 4), method="bounded").x)


def tradeable_quotes(max_spread: float = 0.06) -> pd.DataFrame:
    d = pd.read_parquet(REPORTS / "market_backtest.parquet")
    d = d[(d["yes_ask"] - d["yes_bid"] <= max_spread) & d["p_market"].between(0.03, 0.97)].copy()
    d["lm"] = logit(d["p_market"].to_numpy())
    d["gap"] = logit(d["p_fixed"].to_numpy()) - d["lm"]
    return d


def fit_realization(d: pd.DataFrame | None = None, n_boot: int = 300, k_shrink: float = 3000, seed: int = 0) -> dict:
    d = tradeable_quotes() if d is None else d
    y, off, x = d["y"].to_numpy(), d["lm"].to_numpy(), d["gap"].to_numpy()
    lam = _fit_lambda(y, off, x)
    games = d["game_id"].to_numpy()
    ug = np.unique(games)
    idx_by_game = pd.Series(np.arange(len(d))).groupby(games).apply(np.array)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate(idx_by_game.loc[rng.choice(ug, len(ug))].to_numpy())
        boots.append(_fit_lambda(y[pick], off[pick], x[pick]))
    lo, hi = np.percentile(boots, [10, 90])
    out = dict(pooled=lam, pooled_lo=float(lo), pooled_hi=float(hi), n=int(len(d)), by_stat={}, by_season={})
    for s, g in d.groupby("stat"):
        ls = _fit_lambda(g["y"].to_numpy(), g["lm"].to_numpy(), g["gap"].to_numpy())
        w = len(g) / (len(g) + k_shrink)
        shr = w * ls + (1 - w) * lam
        out["by_stat"][s] = dict(n=int(len(g)), raw=ls, shrunk=shr, lo=shr - (lam - lo))
    for S, g in d.groupby("season"):
        out["by_season"][int(S)] = _fit_lambda(g["y"].to_numpy(), g["lm"].to_numpy(), g["gap"].to_numpy())
    ER_PATH.write_text(json.dumps(out, indent=1))
    return out


def load_realization() -> dict | None:
    return json.loads(ER_PATH.read_text()) if ER_PATH.exists() else None


def robust_prob(p_final, p_mkt, stats, er: dict | None, which: str = "lo", cap: float = 1.0) -> np.ndarray:
    """Leg probability with only the realized part of the blend's edge kept (lambda capped at `cap`:
    never lean further from the market than the blend itself does)."""
    p_final, p_mkt = np.asarray(p_final, float), np.asarray(p_mkt, float)
    if er is None:
        lam = np.full(len(p_final), 0.5)
    else:
        key = "lo" if which == "lo" else "shrunk"
        lam = np.array([er["by_stat"].get(s, {}).get(key, er["pooled_lo" if which == "lo" else "pooled"]) for s in stats])
    lam = np.clip(lam, 0.0, cap)
    return expit(logit(p_mkt) + lam * (logit(p_final) - logit(p_mkt)))


# ----------------------------------------------------------------------------- 2. combo backtest on real prices
def _legs_for_week(w: pd.DataFrame, lam: float, cfg: dict) -> pd.DataFrame:
    c = cfg["combos"]
    w = w[(w["games_career"] >= cfg["risk"]["min_games_history"]) & (w["inj_questionable"].fillna(0) < 1)]
    w = w[(w["p_model"] - w["p_market"]).abs() <= cfg["risk"]["max_disagreement"]]
    pr = expit(w["lm"] + min(lam, 1.0) * w["gap"])
    yes = w.assign(side="yes", p_leg=pr, p_mkt=w["p_market"], hit=w["y"])
    no = w.assign(side="no", p_leg=1 - pr, p_mkt=1 - w["p_market"], hit=1 - w["y"])
    legs = pd.concat([yes, no], ignore_index=True)
    legs = legs[legs["p_leg"].between(c["min_leg_p"], c["max_leg_p"])]
    legs["edge_lo"] = logit(legs["p_leg"].to_numpy()) - logit(legs["p_mkt"].to_numpy())
    legs = legs[legs["edge_lo"] > 0].sort_values("edge_lo", ascending=False)
    legs = legs.groupby(["game_id", "player_id", "stat", "side"], sort=False).head(2).head(c.get("bt_candidates", 40))
    legs = legs.reset_index(drop=True)
    legs["label"] = legs["name"].astype(str) + " " + legs["side"] + " " + (legs["strike"] + 0.5).astype(str) + "+ " + legs["stat"]
    legs["p_model"] = np.where(legs["side"] == "yes", legs["p_model"], 1 - legs["p_model"])
    return legs


def backtest(holds=(0.0, 0.05, 0.10, 0.15), per_size: int = 3, all_stats: bool = False, log=print) -> dict:
    tag = "_allstats" if all_stats else ""
    from . import combos
    cfg = load_config()
    cfg["combos"].update(min_legs=2, max_legs=6, beam=100, draws=20000, min_roi=0.0)
    d = tradeable_quotes()
    pos = pd.read_parquet(DATA / "frame.parquet", columns=["player_id", "position"]).drop_duplicates("player_id")
    d = d.merge(pos, on="player_id", how="left").dropna(subset=["position"])
    if not all_stats:
        d = d[~d["stat"].isin(cfg["combos"].get("not_combinable_stats", []))]
    oos = pd.read_parquet(REPORTS / "oos_predictions.parquet")
    cops = {S: combos.fit_copula(oos, S - 1) for S in sorted(d["season"].unique())}
    rows = []
    weeks = d[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).to_numpy()
    for S, W in weeks:
        prior = d[(d["season"] < S) | ((d["season"] == S) & (d["week"] < W))]
        # walk-forward lambda, shrunk toward 0.5 when there is little history
        lam_hat = _fit_lambda(prior["y"].to_numpy(), prior["lm"].to_numpy(), prior["gap"].to_numpy()) if len(prior) > 500 else 0.5
        lam = (len(prior) * lam_hat + 3000 * 0.5) / (len(prior) + 3000)
        legs = _legs_for_week(d[(d["season"] == S) & (d["week"] == W)], lam, cfg)
        if len(legs) < 6:
            continue
        pr = combos.ComboPricer(legs, cops[S], 20000, seed=int(S * 100 + W))
        res = combos.search(pr, cfg, log=lambda *a: None)
        if res.empty:
            continue
        for k, g in res.groupby("n_legs"):
            used = set()
            for r in g.sort_values("growth", ascending=False).to_dict("records"):
                if used & set(r["legs"]):
                    continue
                used |= set(r["legs"])
                L = pr.legs.iloc[list(r["legs"])]
                rows.append(dict(season=int(S), week=int(W), k=int(k), lam=lam, p_model=r["p_model"], p_fair=r["p_fair"],
                                 p_indep=r["p_indep"], hit=int(L["hit"].all()), n_games=int(L["game_id"].nunique()),
                                 desc=" + ".join(L["label"])))
                if sum(1 for x in rows if x["season"] == S and x["week"] == W and x["k"] == k) >= per_size:
                    break
        log(f"  {S} wk {W:>2}: {len(legs)} legs, lambda {lam:.2f}, {sum(1 for x in rows if x['season'] == S and x['week'] == W)} combos")
    bt = pd.DataFrame(rows)
    bt.to_parquet(REPORTS / f"combo_market_backtest{tag}.parquet")
    summary = summarize(bt, holds)
    (REPORTS / f"combo_market_backtest{tag}.json").write_text(json.dumps(summary, indent=1))
    fit_leg_calibration()
    return summary


CAL_PATH = MODELS / "combo_calibration.json"


def fit_leg_calibration(pseudo: float = 15.0) -> dict:
    """Hit rate / predicted probability by leg count over every real-price backtest combo, shrunk toward 1
    with `pseudo` expected hits. Used only to shrink (never raise) a combo's probability."""
    parts = [pd.read_parquet(p) for p in REPORTS.glob("combo_market_backtest*.parquet")]
    if not parts:
        return {}
    d = pd.concat(parts)
    out = {}
    for k, g in d.groupby("k"):
        f = (g["hit"].sum() + pseudo) / (g["p_model"].sum() + pseudo)
        out[str(int(k))] = dict(n=int(len(g)), predicted=float(g["p_model"].mean()), hit=float(g["hit"].mean()),
                                vs_fair=float(g["hit"].mean() / g["p_fair"].mean()), factor=float(min(1.0, f)))
    CAL_PATH.write_text(json.dumps(out, indent=1))
    return out


def leg_factor(k: int) -> float:
    if not CAL_PATH.exists():
        return 1.0
    cal = json.loads(CAL_PATH.read_text())
    if str(k) in cal:
        return cal[str(k)]["factor"]
    big = [v["factor"] for kk, v in cal.items() if int(kk) <= k]
    return min(big) if big else 1.0


def _price(p_fair, hold):
    c = np.minimum(0.99, p_fair * (1 + hold))
    return c + fee_per_contract(c)


def summarize(bt: pd.DataFrame, holds=(0.0, 0.05, 0.10, 0.15), n_boot: int = 2000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for k, g in bt.groupby("k"):
        r = dict(combos=int(len(g)), weeks=int(g[["season", "week"]].drop_duplicates().shape[0]),
                 p_model=float(g["p_model"].mean()), p_fair=float(g["p_fair"].mean()), p_indep=float(g["p_indep"].mean()),
                 hit_rate=float(g["hit"].mean()), hits=int(g["hit"].sum()), by_hold={})
        wk = g["season"].astype(str) + "-" + g["week"].astype(str)
        for h in holds:
            cost = _price(g["p_fair"].to_numpy(), h)
            pnl = g["hit"].to_numpy() / cost - 1  # per $1 staked
            by = pd.DataFrame({"wk": wk.to_numpy(), "pnl": pnl}).groupby("wk")["pnl"].agg(["sum", "size"])
            idx = rng.integers(0, len(by), (n_boot, len(by)))
            boots = by["sum"].to_numpy()[idx].sum(1) / by["size"].to_numpy()[idx].sum(1)
            eq = np.cumsum(np.r_[0, pnl])
            dd = float((np.maximum.accumulate(eq) - eq).max())
            r["by_hold"][str(h)] = dict(avg_price=float(cost.mean()), pred_roi=float((g["p_model"].to_numpy() / cost - 1).mean()),
                                        roi=float(pnl.mean()), roi_lo=float(np.percentile(boots, 5)),
                                        roi_hi=float(np.percentile(boots, 95)), p_roi_pos=float((boots > 0).mean()),
                                        sd_per_bet=float(pnl.std()), max_dd_units=dd)
        out[int(k)] = r
    return out


# ----------------------------------------------------------------------------- 3. staking plan
def bet_math(q: float, price: float) -> dict:
    """Per-$1 math of one binary contract bought at `price` (fee included) that pays $1 w.p. q."""
    c = price + fee_per_contract(price)
    ev = q / c - 1
    sd = math.sqrt(q * (1 - q)) / c
    f = max(0.0, (q - c) / (1 - c))
    g = lambda frac: q * math.log1p(frac * f * (1 - c) / c) + (1 - q) * math.log1p(-frac * f) if f > 0 else 0.0
    n95 = (1.645 * sd / ev) ** 2 if ev > 0 else float("inf")
    return dict(q=q, price=price, cost=c, payout_x=1 / c, ev_per_dollar=ev, sd_per_dollar=sd, kelly=f,
                growth_q=g(0.25), growth_h=g(0.5), growth_full=g(1.0), bets_to_95pct_sure=n95)


def simulate_season(q_a: float, q_b: float, price: float, bankroll: float, stake_rule, weeks: int, per_week: int,
                    sims: int = 5000, seed: int = 0, ruin: float = 0.5, target: float | None = None) -> dict:
    """Monte-Carlo bankroll paths. The true hit rate of each simulated season is drawn from Beta(q_a, q_b),
    so uncertainty about the edge itself is part of the risk. stake_rule(bankroll) -> $ per combo."""
    rng = np.random.default_rng(seed)
    c = price + fee_per_contract(price)
    qs = rng.beta(q_a, q_b, sims)
    bank = np.full(sims, float(bankroll))
    peak = bank.copy(); mdd = np.zeros(sims); ruined = np.zeros(sims, bool)
    for _ in range(weeks):
        for _ in range(per_week):
            stake = np.minimum(stake_rule(bank), np.maximum(bank, 0))
            win = rng.random(sims) < qs
            bank = bank + np.where(win, stake * (1 / c - 1), -stake)
            peak = np.maximum(peak, bank)
            mdd = np.maximum(mdd, (peak - bank) / peak)
            ruined |= bank < ruin * bankroll
    ret = bank / bankroll - 1
    out = dict(mean_return=float(ret.mean()), median_return=float(np.median(ret)), p5_return=float(np.percentile(ret, 5)),
               p95_return=float(np.percentile(ret, 95)), p_profit=float((ret > 0).mean()),
               median_max_dd=float(np.median(mdd)), p95_max_dd=float(np.percentile(mdd, 95)), p_ruin=float(ruined.mean()))
    if target is not None:
        out["p_target"] = float((ret >= target).mean())
    return out


def plan(bankroll: float, weeks: int, per_week: int, target: float, hold: float = 0.10,
         summary: dict | None = None, frac: float = 0.25, all_stats: bool = False) -> pd.DataFrame:
    """One row per leg count: what to stake and what to expect, from the real-price backtest."""
    p = REPORTS / f"combo_market_backtest{'_allstats' if all_stats else ''}.json"
    summary = summary or (json.loads(p.read_text()) if p.exists() else None)
    if not summary:
        raise SystemExit("run `python -m propbot combo-backtest-market` first")
    rows = []
    for k, s in sorted(summary.items(), key=lambda t: int(t[0])):
        h = s["by_hold"].get(str(hold)) or s["by_hold"][min(s["by_hold"], key=lambda x: abs(float(x) - hold))]
        price = h["avg_price"] - fee_per_contract(h["avg_price"])
        n = s["combos"]
        # hit rate: the LOWER of the backtest's own (small samples, it ran hot) and what the model predicted,
        # with a Beta spread as wide as the backtest sample, so the edge itself is uncertain in the simulation
        q_hat = min((s["hits"] + 1) / (n + 2), s["p_model"])
        a, b = q_hat * (n + 2), (1 - q_hat) * (n + 2)
        m = bet_math(q_hat, price)
        kelly_stake = lambda bank, fr=frac: fr * m["kelly"] * bank
        need = target * bankroll / (weeks * per_week * m["ev_per_dollar"]) if m["ev_per_dollar"] > 0 else float("nan")
        sim_k = simulate_season(a, b, price, bankroll, kelly_stake, weeks, per_week, target=target)
        sim_t = simulate_season(a, b, price, bankroll, lambda bank: np.full_like(bank, need), weeks, per_week, target=target) \
            if need == need else {}
        rows.append(dict(
            legs=int(k), combos_tested=n, hit_rate=q_hat, price=price, payout_x=m["payout_x"],
            ev_per_dollar=m["ev_per_dollar"], roi_ci90=f"{h['roi_lo']:+.0%}..{h['roi_hi']:+.0%}", sd_per_dollar=m["sd_per_dollar"],
            kelly=m["kelly"], stake_frac_kelly=frac * m["kelly"] * bankroll,
            kelly_season_ret=sim_k["median_return"], kelly_p_profit=sim_k["p_profit"], kelly_p_target=sim_k.get("p_target"),
            kelly_p95_dd=sim_k["p95_max_dd"], kelly_p_ruin=sim_k["p_ruin"],
            stake_for_target=need, target_x_kelly=need / (m["kelly"] * bankroll) if m["kelly"] > 0 else float("inf"),
            target_p_target=sim_t.get("p_target"), target_p_ruin=sim_t.get("p_ruin"), target_p95_dd=sim_t.get("p95_max_dd"),
            bets_to_95pct_sure=m["bets_to_95pct_sure"]))
    return pd.DataFrame(rows)


def leg_count_math(p_leg: float, edge: float, hold: float = 0.10, rho: float = 0.0, ks=range(1, 9),
                   bankroll: float = 500, frac: float = 0.25, n_bets: int = 24) -> pd.DataFrame:
    """Same-quality legs, different leg counts. Each leg: our probability p_leg, the market's p_leg/(1+edge).
    Legs share pairwise latent correlation rho (0 = different games). The combo is quoted at the market's own
    correlated fair price x (1 + hold). Shows how EV, variance, Kelly, growth and the number of bets needed
    to see the edge scale with k."""
    from scipy.stats import multivariate_normal, norm
    p_mkt = p_leg / (1 + edge)
    rows = []
    for k in ks:
        def joint(p):
            if k == 1 or rho == 0:
                return p ** k
            C = np.full((k, k), rho) + (1 - rho) * np.eye(k)
            z = norm.ppf(1 - p)
            return float(multivariate_normal(mean=np.zeros(k), cov=C).cdf(-np.full(k, z)))
        q, fair = joint(p_leg), joint(p_mkt)
        price = min(0.99, fair * (1 + (hold if k > 1 else 0.0)))
        m = bet_math(q, price)
        stake = frac * m["kelly"] * bankroll
        rows.append(dict(legs=k, p_win=q, price=price, payout_x=m["payout_x"], ev_per_dollar=m["ev_per_dollar"],
                         sd_per_dollar=m["sd_per_dollar"], kelly=m["kelly"], stake=stake,
                         exp_profit_per_bet=stake * m["ev_per_dollar"], growth_per_bet_q=m["growth_q"],
                         p_ahead_after_n=float(1 - norm.cdf(0, n_bets * m["ev_per_dollar"], math.sqrt(n_bets) * m["sd_per_dollar"])),
                         bets_to_95pct_sure=m["bets_to_95pct_sure"]))
    return pd.DataFrame(rows)


def stake_table(q: float, price: float, stakes=(5, 10, 25, 50, 100)) -> pd.DataFrame:
    """Money in -> money out for one combo."""
    m = bet_math(q, price)
    c = m["cost"]
    return pd.DataFrame([dict(stake=s, contracts=math.floor(s / c), payout_if_win=math.floor(s / c),
                              profit_if_win=math.floor(s / c) - s, loss_if_lose=-s, expected_profit=s * m["ev_per_dollar"],
                              p_win=q) for s in stakes])


# ----------------------------------------------------------------------------- 4. policy + paper ledger
SCHEMA = """
CREATE TABLE IF NOT EXISTS combos(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mode TEXT, legs TEXT, desc TEXT, n_legs INTEGER,
  p_model REAL, p_robust REAL, p_fair REAL, price REAL, max_price REAL, contracts INTEGER, stake REAL,
  status TEXT DEFAULT 'open', payout REAL, pnl REAL, settled_ts TEXT);
"""


def _db():
    from .ledger import Ledger
    L = Ledger()
    L.con.executescript(SCHEMA)
    return L.con


def choose(res: dict, cfg: dict, bankroll: float, open_stake: float = 0.0) -> pd.DataFrame:
    """Policy: robust re-price, gate, size. `res` is combos.suggest() output."""
    c = cfg["combos"]
    er = load_realization()
    L = res["legs"].copy()
    L["p_robust"] = robust_prob(np.where(L["side"] == "yes", L["p_final"], 1 - L["p_final"]), L["p_mkt"], L["stat"], er, "lo")
    pr = res["pricer"]
    out = []
    budget = c.get("max_weekly_combo_frac", 0.05) * bankroll - open_stake
    for r in res["combos"].sort_values("growth", ascending=False).to_dict("records"):
        if r["n_legs"] not in c.get("allowed_legs", [2, 3, 4, 5]):
            continue
        idx = list(r["legs"])
        sub = L.iloc[idx]
        # robust joint probability: same correlation model, legs at p_robust
        p_rob = _joint_with(pr, idx, sub["p_robust"].to_numpy()) * leg_factor(r["n_legs"])
        m = bet_math(p_rob, r["est_price"])
        if m["ev_per_dollar"] < c.get("min_roi", 0.10) or m["kelly"] <= 0:
            continue
        stake = min(c.get("combo_kelly_fraction", 0.10) * m["kelly"] * bankroll, c.get("max_stake_per_combo", 10.0), budget)
        n = math.floor(stake / m["cost"])
        if n < 1:
            continue
        budget -= n * m["cost"]
        out.append(dict(legs=json.dumps([[t, s] for t, s in zip(sub["ticker"], sub["side"])]), desc=r["desc"], n_legs=r["n_legs"],
                        p_model=r["p_model"], p_robust=p_rob, p_fair=r["p_fair"], price=r["est_price"], max_price=r["max_price"],
                        contracts=n, stake=n * m["cost"], ev=m["ev_per_dollar"]))
        if len(out) >= c.get("max_combos_per_run", 3):
            break
    return pd.DataFrame(out)


def _joint_with(pr, idx, p_legs) -> float:
    """Joint probability of legs `idx` at probabilities `p_legs`, reusing the pricer's latent draws."""
    from .combos import simulate
    L = pr.legs.iloc[idx]
    U = simulate(pr.vars, pr.cop, 20000, seed=11)[:, L["var"].to_numpy()]
    yes = (L["side"] == "yes").to_numpy()
    h = np.where(yes, U >= 1 - p_legs, U < p_legs)
    return float(h.all(axis=1).mean())


def record(picks: pd.DataFrame, mode: str = "paper") -> int:
    import datetime as dt
    con = _db()
    for p in picks.to_dict("records"):
        con.execute("INSERT INTO combos(ts, mode, legs, desc, n_legs, p_model, p_robust, p_fair, price, max_price, contracts, stake)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (dt.datetime.now(dt.timezone.utc).isoformat(), mode, p["legs"], p["desc"], p["n_legs"], p["p_model"],
                     p["p_robust"], p["p_fair"], p["price"], p["max_price"], p["contracts"], p["stake"]))
    con.commit()
    return len(picks)


def open_combos(mode: str = "paper") -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM combos WHERE mode=? AND status='open'", _db(), params=[mode])


def settle(mode: str = "paper", log=print) -> int:
    import datetime as dt
    from .kalshi import Kalshi
    con, k, n = _db(), Kalshi("prod"), 0
    for t in open_combos(mode).to_dict("records"):
        legs = json.loads(t["legs"])
        res = []
        for tk, side in legs:
            m = k.market(tk)
            r = m.get("result")
            res.append(None if r not in ("yes", "no") else (r == side))
        if any(x is False for x in res) or all(x is True for x in res):
            win = all(x is True for x in res)
            payout = t["contracts"] * (1.0 if win else 0.0)
            con.execute("UPDATE combos SET status='settled', payout=?, pnl=?, settled_ts=? WHERE id=?",
                        (payout, payout - t["stake"], dt.datetime.now(dt.timezone.utc).isoformat(), t["id"]))
            n += 1
    con.commit()
    log(f"  settled {n} {mode} combos")
    return n


def status(mode: str = "paper") -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM combos WHERE mode=?", _db(), params=[mode])
