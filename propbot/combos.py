"""Combo (parlay) builder for Kalshi multi-leg markets.

A Kalshi combo pays $1 only if every leg resolves YES. Legs inside one game are correlated (a QB's
yards and his WR1's yards move together; two RBs on one team split carries), so the probability of a
combo is NOT the product of the legs.

1. Correlation model (Gaussian copula)
   Every (player, stat) in a game gets a latent standard-normal variable. A leg "X N+" hits when the
   latent is above the level matching P(X >= N); all rungs of one ladder share the same latent, so
   "JSN 80+" can only hit if "JSN 60+" does. Latent correlations are estimated from out-of-sample
   model residuals (actual - projection, as normal scores) for every pair of (role, stat) within the same
   player, the same team, or opposing teams, and shrunk toward 0 when the sample is small.

2. Pricing
   Monte-Carlo the whole slate once (default 40k draws). A combo's probability is the share of draws
   in which all its legs hit. Two probabilities are computed for every combo:
     p_model  legs at our blended probability (model shrunk toward market),
     p_fair   legs at the market's own mid-prices, same correlations -> what a market maker who prices
              correlation correctly would charge before margin.
   The expected quote is p_fair * (1 + hold); "max price" is the most you should accept.

3. Search
   Beam search over legs (YES = over, NO = under) choosing the number of legs and the rung of each
   ladder (e.g. JSN 60+ vs JSN 80+) to maximise expected log bankroll growth at fractional Kelly.
   Report the best combo for each leg count, and for each leg what swapping to a neighbouring rung does.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import DATA, MODELS, REPORTS, STATS
from .edge import fee_per_contract, logit

POSG = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
COPULA = MODELS / "copula.json"
MIN_N = 300           # pairs needed before a fine-grained correlation is trusted on its own
SHRINK_N = 400        # rho_shrunk = rho * n / (n + SHRINK_N) toward the coarser estimate


# ----------------------------------------------------------------------------- correlation model
def _role(posg: str, stat: str) -> str:
    return f"{posg}:{stat}"


def pair_keys(rel: str, ra: str, rb: str) -> tuple[str, str]:
    """(fine, coarse) keys for a pair of (role) strings with relation self/team/opp."""
    a, b = sorted([ra, rb])
    sa, sb = sorted([ra.split(":")[1], rb.split(":")[1]])
    return f"{rel}|{a}|{b}", f"{rel}|*{sa}|*{sb}"


def _normal_scores(v: pd.DataFrame) -> np.ndarray:
    r = (v["actual"] - v["mu"]) / np.sqrt(v["mu"].clip(lower=0) + 1)
    rank = r.groupby([v["stat"], v["posg"]]).rank(method="average")
    n = r.groupby([v["stat"], v["posg"]]).transform("size")
    return norm.ppf(rank / (n + 1))


def _variables(oos: pd.DataFrame) -> pd.DataFrame:
    v = oos.drop_duplicates(["game_id", "player_id", "stat"])[
        ["season", "game_id", "player_id", "position", "team", "stat", "mu", "actual"]].copy()
    v["posg"] = v["position"].map(POSG)
    # keep the players Kalshi actually lists: top projections per team / stat
    v["rk"] = v.groupby(["game_id", "team", "stat"])["mu"].rank(ascending=False, method="first")
    lim = v["stat"].map(lambda s: 1 if STATS[s].get("starter_qb") else 4)
    v = v[v["rk"] <= lim].copy()
    v["z"] = _normal_scores(v)
    v["role"] = v["posg"] + ":" + v["stat"]
    return v.drop(columns="rk")


def _pairs(v: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "player_id", "team", "role", "stat", "z"]
    a = v[cols].reset_index(drop=True)
    a["i"] = np.arange(len(a))
    p = a.merge(a, on="game_id", suffixes=("_a", "_b"))
    p = p[p["i_a"] < p["i_b"]]
    rel = np.where(p["player_id_a"] == p["player_id_b"], "self", np.where(p["team_a"] == p["team_b"], "team", "opp"))
    ra, rb = p["role_a"].to_numpy(), p["role_b"].to_numpy()
    lo, hi = np.where(ra <= rb, ra, rb), np.where(ra <= rb, rb, ra)
    sa, sb = p["stat_a"].to_numpy(), p["stat_b"].to_numpy()
    slo, shi = np.where(sa <= sb, sa, sb), np.where(sa <= sb, sb, sa)
    return pd.DataFrame({"fine": pd.Series(rel) + "|" + lo + "|" + hi,
                         "coarse": pd.Series(rel) + "|*" + slo + "|*" + shi,
                         "za": p["z_a"].to_numpy(), "zb": p["z_b"].to_numpy()})


def _corr_table(p: pd.DataFrame, key: str) -> pd.DataFrame:
    g = p.groupby(key)
    m = g.agg(n=("za", "size"), ma=("za", "mean"), mb=("zb", "mean"))
    m["sab"] = g.apply(lambda d: float((d["za"] * d["zb"]).mean()), include_groups=False)
    m["va"] = g["za"].var(ddof=0); m["vb"] = g["zb"].var(ddof=0)
    m["rho"] = (m["sab"] - m["ma"] * m["mb"]) / np.sqrt(m["va"] * m["vb"])
    return m[["n", "rho"]]


def fit_copula(oos: pd.DataFrame, max_season: int | None = None) -> dict:
    v = _variables(oos if max_season is None else oos[oos["season"] <= max_season])
    p = _pairs(v)
    coarse = _corr_table(p, "coarse")
    fine = _corr_table(p, "fine")
    rho = {}
    for k, r in coarse.iterrows():
        rho[k] = float(np.clip(r["rho"] * r["n"] / (r["n"] + SHRINK_N), -0.95, 0.95))
    for k, r in fine.iterrows():
        rel, a, b = k.split("|")
        base = rho.get(pair_keys(rel, a, b)[1], 0.0)
        w = r["n"] / (r["n"] + SHRINK_N)
        rho[k] = float(np.clip(w * r["rho"] + (1 - w) * base, -0.95, 0.95))
    return dict(rho=rho, n={k: int(v) for k, v in pd.concat([coarse["n"], fine["n"]]).items()},
                max_season=None if max_season is None else int(max_season))


def save_copula(c: dict):
    COPULA.write_text(json.dumps(c))


def load_copula() -> dict:
    return json.loads(COPULA.read_text()) if COPULA.exists() else dict(rho={}, n={})


def corr_matrix(vars_: pd.DataFrame, cop: dict) -> np.ndarray:
    """vars_: rows (game_id, player_id, team, posg, stat). Block-diagonal by game, PSD-repaired."""
    n = len(vars_)
    C = np.eye(n)
    g, pl, tm = vars_["game_id"].to_numpy(), vars_["player_id"].to_numpy(), vars_["team"].to_numpy()
    roles = (vars_["posg"] + ":" + vars_["stat"]).to_numpy()
    rho = cop["rho"]
    for i in range(n):
        for j in range(i + 1, n):
            if g[i] != g[j]:
                continue
            rel = "self" if pl[i] == pl[j] else ("team" if tm[i] == tm[j] else "opp")
            fine, coarse = pair_keys(rel, roles[i], roles[j])
            r = rho.get(fine, rho.get(coarse, 0.7 if rel == "self" else 0.0))
            C[i, j] = C[j, i] = r
    w, V = np.linalg.eigh(C)
    if w.min() < 1e-6:  # nearest-PSD repair, then back to unit diagonal
        C = (V * np.clip(w, 1e-6, None)) @ V.T
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
    return C


def simulate(vars_: pd.DataFrame, cop: dict, n: int = 40000, seed: int = 7) -> np.ndarray:
    """Uniform latents U (n_draws x n_vars). Larger U = bigger stat."""
    rng = np.random.default_rng(seed)
    U = np.empty((n, len(vars_)), dtype=np.float32)
    for _, idx in vars_.groupby("game_id").indices.items():
        C = corr_matrix(vars_.iloc[idx], cop)
        L = np.linalg.cholesky(C)
        U[:, idx] = norm.cdf(rng.standard_normal((n, len(idx))) @ L.T)
    return U


# ----------------------------------------------------------------------------- legs
ELIG_CACHE = DATA / "combo_eligible.json"


def eligible_events(log=print) -> dict | None:
    """Event tickers Kalshi lets you put in an NFL combo, from the exchange's own multivariate
    collections ({event_ticker: yes_only}). Cached for 6 hours. None if the API is unreachable."""
    import time
    if ELIG_CACHE.exists() and time.time() - ELIG_CACHE.stat().st_mtime < 6 * 3600:
        return json.loads(ELIG_CACHE.read_text())
    try:
        from .kalshi import Kalshi
        k = Kalshi("prod")
        out, cursor = {}, None
        for _ in range(20):
            r = k._req("GET", "/multivariate_event_collections",
                       params={"status": "open", "limit": 200, **({"cursor": cursor} if cursor else {})})
            for col in r.get("multivariate_contracts", []):
                for e in col.get("associated_events", []):
                    if e["ticker"].startswith("KXNFL"):
                        out[e["ticker"]] = bool(e.get("is_yes_only")) or out.get(e["ticker"], False)
            cursor = r.get("cursor")
            if not cursor:
                break
        ELIG_CACHE.write_text(json.dumps(out))
        return out
    except Exception as e:
        log(f"  [combos] could not read combo collections ({type(e).__name__}); eligibility from config only")
        return None


def candidate_legs(px: pd.DataFrame, cfg: dict, log=print) -> pd.DataFrame:
    """Turn a priced scan (bot.scan / last_scan.parquet) into YES/NO legs with edge vs the market."""
    c = cfg.get("combos", {})
    d = px.dropna(subset=["p_final", "p_market"]).copy()
    # Kalshi-specific: only markets the combo builder accepts. The config list mirrors the Help Center
    # ("receptions, receiving yards and rushing yards ladders cannot be added to combos"); the exchange's
    # collection list is applied on top when reachable.
    d = d[~d["stat"].isin(c.get("not_combinable_stats", []))]
    d = d[~d["series"].str.endswith("PROT")] if "series" in d.columns else d
    yes_only = set()
    if c.get("require_eligible", True):
        el = eligible_events(log)
        if el is not None and "event" in d.columns:
            d = d[d["event"].isin(el)]
            yes_only = {e for e, y in el.items() if y}
    d = d[d["minutes_to_kickoff"] > cfg["risk"]["min_minutes_to_kickoff"]]
    d = d[d["games_career"] >= cfg["risk"]["min_games_history"]]
    if cfg["risk"].get("skip_questionable", True):
        d = d[d["inj_questionable"].fillna(0) < 1]
    d = d[d["spread"] <= c.get("max_leg_spread", 0.10)]
    # a big model/market gap usually means news the model hasn't seen (injury, role change), not edge
    d = d[(d["p_model"] - d["p_market"]).abs() <= c.get("max_disagreement", cfg["risk"]["max_disagreement"])]
    d = d[~d["stat"].isin(c.get("exclude_stats", []))]
    yes = d.assign(side="yes", p_leg=d["p_final"], p_mkt=d["p_market"])
    no = d.assign(side="no", p_leg=1 - d["p_final"], p_mkt=1 - d["p_market"])
    if "event" in no.columns and yes_only:
        no = no[~no["event"].isin(yes_only)]
    legs = pd.concat([yes, no], ignore_index=True)
    if not c.get("allow_no", True):
        legs = legs[legs["side"] == "yes"]
    legs = legs[legs["p_leg"].between(c.get("min_leg_p", 0.25), c.get("max_leg_p", 0.93))]
    legs["edge_lo"] = logit(legs["p_leg"].to_numpy()) - logit(legs["p_mkt"].to_numpy())
    legs = legs[legs["edge_lo"] >= c.get("min_leg_edge", 0.0)]
    # keep the best few rungs per ladder/side, then the best legs overall
    legs = legs.sort_values("edge_lo", ascending=False)
    legs = legs.groupby(["game_id", "player_id", "stat", "side"], sort=False).head(c.get("rungs_per_ladder", 4))
    legs = legs.head(c.get("max_candidates", 70)).reset_index(drop=True)
    legs["label"] = [_label(r) for r in legs.to_dict("records")]
    return legs


def _label(r: dict) -> str:
    n = r["strike"] + 0.5
    name = {"pass_yds": "pass yds", "pass_tds": "pass TD", "pass_att": "pass att", "pass_comp": "completions",
            "pass_int": "INT", "rush_yds": "rush yds", "rush_att": "rush att", "rec": "rec", "rec_yds": "rec yds",
            "rr_yds": "rush+rec yds", "td": "TD"}[r["stat"]]
    s = f"{r['name']} {n:g}+ {name}"
    return s if r["side"] == "yes" else f"{r['name']} UNDER {n:g} {name} (NO on {n:g}+)"


# ----------------------------------------------------------------------------- search
def _growth(q, c, frac):
    """Expected log growth of betting a `frac`-Kelly stake on a binary bet won w.p. q at all-in cost c."""
    if c <= 0 or c >= 1 or q <= c:
        return 0.0, 0.0
    f = frac * (q - c) / (1 - c)
    return float(q * math.log1p(f * (1 - c) / c) + (1 - q) * math.log1p(-f)), f


class ComboPricer:
    def __init__(self, legs: pd.DataFrame, cop: dict, n_draws: int = 40000, seed: int = 7):
        self.legs = legs.reset_index(drop=True)
        v = self.legs[["game_id", "player_id", "team", "position", "stat"]].drop_duplicates(["game_id", "player_id", "stat"])
        v = v.assign(posg=v["position"].map(POSG)).reset_index(drop=True)
        self.vars = v
        vid = {k: i for i, k in enumerate(zip(v["game_id"], v["player_id"], v["stat"]))}
        self.legs["var"] = [vid[k] for k in zip(self.legs["game_id"], self.legs["player_id"], self.legs["stat"])]
        self.cop = cop
        U = simulate(v, cop, n_draws, seed) if len(v) else np.empty((n_draws, 0))
        u = U[:, self.legs["var"].to_numpy()]
        yes = (self.legs["side"] == "yes").to_numpy()
        pm, pk = self.legs["p_leg"].to_numpy(), self.legs["p_mkt"].to_numpy()
        # YES on "N+" hits when U >= 1 - P(N+); NO hits when U < 1 - P(N+) (P(N+) = 1 - p_leg for NO)
        self.hit = np.where(yes, u >= 1 - pm, u < pm).T.copy()          # model marginals
        self.hit_mkt = np.where(yes, u >= 1 - pk, u < pk).T.copy()      # market marginals

    def rho(self, i: int, j: int) -> float:
        """Latent correlation between legs i and j (sign-flipped for NO legs)."""
        a, b = self.legs.iloc[i], self.legs.iloc[j]
        if a["var"] == b["var"]:
            r = 1.0
        elif a["game_id"] != b["game_id"]:
            r = 0.0
        else:
            rel = "self" if a["player_id"] == b["player_id"] else ("team" if a["team"] == b["team"] else "opp")
            ra, rb = f"{POSG.get(a['position'])}:{a['stat']}", f"{POSG.get(b['position'])}:{b['stat']}"
            fine, coarse = pair_keys(rel, ra, rb)
            r = self.cop["rho"].get(fine, self.cop["rho"].get(coarse, 0.7 if rel == "self" else 0.0))
        return r * (1 if a["side"] == b["side"] else -1)

    def price(self, idx: tuple[int, ...]) -> tuple[float, float]:
        h = self.hit[idx[0]].copy(); hm = self.hit_mkt[idx[0]].copy()
        for i in idx[1:]:
            h &= self.hit[i]; hm &= self.hit_mkt[i]
        return float(h.mean()), float(hm.mean())


def evaluate(p_model: float, p_fair: float, legs: pd.DataFrame, cfg: dict) -> dict:
    c = cfg.get("combos", {})
    hold = c.get("hold", 0.10)
    est = min(0.99, p_fair * (1 + hold))
    cost = est + fee_per_contract(est)
    g, f = _growth(p_model, cost, cfg["risk"]["kelly_fraction"])
    min_roi = c.get("min_roi", 0.10)
    # highest price p with p + fee(p) <= p_model / (1 + min_roi)
    target = p_model / (1 + min_roi)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mid + fee_per_contract(mid) <= target else (lo, mid)
    return dict(p_model=p_model, p_fair=p_fair, p_indep=float(np.prod(legs["p_mkt"])), est_price=est,
                max_price=lo, ev=p_model - cost, roi=(p_model - cost) / cost if cost > 0 else np.nan,
                growth=g, kelly=f)


def search(pr: ComboPricer, cfg: dict, log=print) -> pd.DataFrame:
    c = cfg.get("combos", {})
    kmin, kmax = c.get("min_legs", 2), c.get("max_legs", 6)
    beam_w, min_p = c.get("beam", 250), c.get("min_combo_p", 0.04)
    same_game = c.get("same_game_only", False)
    L = pr.legs
    ladder = list(zip(L["game_id"], L["player_id"], L["stat"]))
    games = L["game_id"].to_numpy()
    players = L["player_id"].to_numpy()
    max_game, max_player = c.get("max_legs_per_game", 4), c.get("max_legs_per_player", 2)
    max_rho = c.get("max_pair_rho", 0.85)

    def ok(idx, j):
        if sum(games[i] == games[j] for i in idx) >= max_game or sum(players[i] == players[j] for i in idx) >= max_player:
            return False
        return all(abs(pr.rho(i, j)) <= max_rho for i in idx)
    n = len(L)
    seen, results = set(), []
    beam = [((i,), *pr.price((i,))) for i in range(n)]
    for k in range(2, kmax + 1):
        cand = []
        for idx, _, _ in beam:
            used = {ladder[i] for i in idx}
            for j in range(n):
                if j in idx or ladder[j] in used or (same_game and games[j] != games[idx[0]]) or not ok(idx, j):
                    continue
                key = tuple(sorted(idx + (j,)))
                if key in seen:
                    continue
                seen.add(key)
                pm, pf = pr.price(key)
                if pm < min_p or pf <= 0:
                    continue
                cand.append((key, pm, pf))
        if not cand:
            break
        scored = []
        for key, pm, pf in cand:
            e = evaluate(pm, pf, L.iloc[list(key)], cfg)
            scored.append((key, pm, pf, e))
        # keep the beam diverse: best by growth and best by edge ratio
        by_g = sorted(scored, key=lambda t: -t[3]["growth"])[:beam_w]
        by_r = sorted(scored, key=lambda t: -(t[1] / t[2]))[:beam_w // 2]
        keep = {t[0]: t for t in by_g + by_r}
        beam = [(t[0], t[1], t[2]) for t in keep.values()]
        if k >= kmin:
            results += [dict(legs=t[0], n_legs=k, **t[3]) for t in keep.values()]
        log(f"  {k} legs: {len(cand):>6} combos priced, best growth {by_g[0][3]['growth']:.5f}")
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results).drop_duplicates("legs")
    out = out[(out["ev"] > 0) & (out["roi"] >= c.get("min_roi", 0.10))]
    return out.sort_values("growth", ascending=False).reset_index(drop=True)


def _ece_by_stat() -> dict:
    p = REPORTS / "backtest.json"
    if not p.exists():
        return {}
    return {k: v.get("ece_cal", 0.01) for k, v in json.loads(p.read_text()).items()}


def risk_report(pr: ComboPricer, combo: tuple, cfg: dict, n_sims: int = 4000, seed: int = 3) -> dict:
    """Parameters 176-200 and 226-250 for one combo: failure counts, pairwise correlations and conditional
    hit rates, structure / concentration, calibration band, pessimistic EV, Kelly stake and drawdown."""
    from .edge import blend
    L = pr.legs.iloc[list(combo)]
    H = pr.hit[list(combo)]
    hits = H.sum(axis=0)
    k = len(combo)
    p_all = float((hits == k).mean())
    out = dict(n_legs=k, p_all_hit=p_all, p_all_fail=float((hits == 0).mean()), p_ge1_fail=1 - p_all,
               p_ge2_fail=float((hits <= k - 2).mean()), p_indep_model=float(np.prod(L["p_leg"])))
    # pairwise: latent rho and P(B | A) from the simulation
    pairs = []
    for a, b in itertools.combinations(range(k), 2):
        pa = H[a].mean()
        pairs.append(dict(a=L.iloc[a]["label"], b=L.iloc[b]["label"], rho=pr.rho(combo[a], combo[b]),
                          p_b_given_a=float((H[a] & H[b]).mean() / pa) if pa > 0 else np.nan,
                          p_b=float(H[b].mean())))
    out["pairs"] = pairs
    out["max_abs_rho"] = max((abs(p["rho"]) for p in pairs), default=0.0)
    out["mean_rho"] = float(np.mean([p["rho"] for p in pairs])) if pairs else 0.0
    # structure
    teams = L.groupby("game_id")["team"].agg(lambda t: t.value_counts().max())
    out.update(n_games=int(L["game_id"].nunique()), max_legs_one_game=int(L["game_id"].value_counts().max()),
               max_same_team=int(teams.max()), max_legs_one_player=int(L["player_id"].value_counts().max()),
               n_opposing_pairs=sum(1 for a, b in itertools.combinations(range(k), 2)
                                    if L.iloc[a]["game_id"] == L.iloc[b]["game_id"] and L.iloc[a]["team"] != L.iloc[b]["team"]),
               shared_game_legs=int(k - L["game_id"].nunique()))
    # calibration band: every leg moved by 2x its stat's historical calibration error
    ece = _ece_by_stat()
    d = np.array([2 * ece.get(s_, 0.01) for s_ in L["stat"]])
    u = np.random.default_rng(seed)
    idx = list(combo)
    p_leg = L["p_leg"].to_numpy()
    yes = (L["side"] == "yes").to_numpy()
    U = None
    try:
        U_vars = simulate(pr.vars, pr.cop, n_sims, seed)
        U = U_vars[:, L["var"].to_numpy()]
    except Exception:
        pass
    def joint(pl):
        if U is None:
            return float(np.prod(pl))
        h = np.where(yes, U >= 1 - pl, U < pl)
        return float(h.all(axis=1).mean())
    out["p_lo"] = joint(np.clip(p_leg - d, 0.01, 0.99))
    out["p_hi"] = joint(np.clip(p_leg + d, 0.01, 0.99))
    out["mc_se"] = float(np.sqrt(p_all * (1 - p_all) / H.shape[1]))
    # pessimistic: legs shrunk to 90% market (instead of the configured weight)
    p_model_raw = np.where(yes, L["p_model"], 1 - L["p_model"])
    pess = blend(p_model_raw, L["p_mkt"].to_numpy(), 0.9)
    out["p_pessimistic"] = joint(np.clip(pess, 0.01, 0.99))
    return out


def portfolio_drawdown(res: pd.DataFrame, bankroll: float, weeks: int = 17, sims: int = 2000, seed: int = 5) -> dict:
    """Bet the shortlist every week for a season (independent across combos/weeks): bankroll paths."""
    if res.empty:
        return {}
    rng = np.random.default_rng(seed)
    q, c, st = res["p_model"].to_numpy(), res["est_price"].to_numpy(), res["stake"].to_numpy()
    fee = np.array([fee_per_contract(x) for x in c])
    n = np.floor(st / (c + fee))
    pnl_win, pnl_loss = n * (1 - c - fee), -n * (c + fee)
    wins = rng.random((sims, weeks, len(q))) < q
    week = np.where(wins, pnl_win, pnl_loss).sum(axis=2)
    eq = bankroll + np.cumsum(week, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.full((sims, 1), bankroll), eq], axis=1), axis=1)[:, 1:]
    dd = ((peak - eq) / peak).max(axis=1)
    return dict(exp_season_pnl=float(week.sum(axis=1).mean()), p_losing_season=float((week.sum(axis=1) < 0).mean()),
                median_max_drawdown=float(np.median(dd)), p95_max_drawdown=float(np.percentile(dd, 95)))


def pick(res: pd.DataFrame, n: int = 10, max_reuse: int = 2) -> pd.DataFrame:
    """Diverse shortlist: greedy by growth, each leg used in at most `max_reuse` combos."""
    used, rows = {}, []
    for r in res.to_dict("records"):
        if any(used.get(i, 0) >= max_reuse for i in r["legs"]):
            continue
        rows.append(r)
        for i in r["legs"]:
            used[i] = used.get(i, 0) + 1
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def rung_alternatives(pr: ComboPricer, legs_all: pd.DataFrame, combo: tuple, cfg: dict) -> list[dict]:
    """For each leg, what the combo looks like with a different rung (or side) of the same ladder."""
    L = pr.legs
    out = []
    for pos, i in enumerate(combo):
        li = L.iloc[i]
        sib = L[(L["game_id"] == li["game_id"]) & (L["player_id"] == li["player_id"]) & (L["stat"] == li["stat"])
                & (L.index != i)]
        for j, lj in sib.iterrows():
            alt = tuple(sorted(combo[:pos] + (j,) + combo[pos + 1:]))
            pm, pf = pr.price(alt)
            e = evaluate(pm, pf, L.iloc[list(alt)], cfg)
            out.append(dict(swap_out=li["label"], swap_in=lj["label"], p_model=pm, roi=e["roi"], growth=e["growth"]))
    return out


def suggest(px: pd.DataFrame, cfg: dict, top: int = 10, log=print) -> dict:
    cop = load_copula()
    if not cop["rho"]:
        log("  [combos] no copula fitted yet (run `train` or `fit-copula`); legs treated as independent")
    legs = candidate_legs(px, cfg, log)
    log(f"  [combos] {len(legs)} candidate legs from {legs['game_id'].nunique() if len(legs) else 0} games")
    if len(legs) < 2:
        return dict(legs=legs, combos=pd.DataFrame(), best_by_size=pd.DataFrame(), pricer=None)
    pr = ComboPricer(legs, cop, cfg.get("combos", {}).get("draws", 40000))
    res = search(pr, cfg, log)
    if res.empty:
        return dict(legs=pr.legs, combos=res, best_by_size=res, pricer=pr)
    best = res.sort_values("growth", ascending=False).drop_duplicates("n_legs").sort_values("n_legs")
    bank = cfg["risk"]["bankroll"]
    cap = cfg.get("combos", {}).get("max_stake_per_combo", 10.0)
    for d in (res, best):
        d["stake"] = np.minimum(bank * d["kelly"], cap).round(2)
        d["desc"] = [" + ".join(pr.legs.loc[list(t), "label"]) for t in d["legs"]]
    short = pick(res, top)
    short["risk"] = [risk_report(pr, t, cfg) for t in short["legs"]]
    return dict(legs=pr.legs, combos=short, best_by_size=best, pricer=pr,
                portfolio=portfolio_drawdown(short, bank))


# ----------------------------------------------------------------------------- honest check
def backtest_copula(oos: pd.DataFrame, test_from: int, n_combos: int = 20000, legs_per: tuple = (2, 3, 4),
                    seed: int = 0, log=print) -> dict:
    """Fit correlations on seasons < test_from, then draw random same-game combos from later seasons'
    out-of-sample ladder probabilities and compare joint log loss: independence vs copula."""
    cop = fit_copula(oos, test_from - 1)
    te = oos[(oos["season"] >= test_from) & oos["p_cal"].between(0.25, 0.85)].copy()
    v = _variables(oos[oos["season"] >= test_from])[["game_id", "player_id", "stat"]]
    te = te.merge(v, on=["game_id", "player_id", "stat"])
    te["posg"] = te["position"].map(POSG)
    rng = np.random.default_rng(seed)
    by_game = {g: d.reset_index(drop=True) for g, d in te.groupby("game_id")}
    gids = [g for g, d in by_game.items() if d[["player_id", "stat"]].drop_duplicates().shape[0] >= max(legs_per)]
    rows = []
    for _ in range(n_combos):
        g = gids[rng.integers(len(gids))]
        d = by_game[g]
        k = int(rng.choice(legs_per))
        lad = d.drop_duplicates(["player_id", "stat"]).sample(k, random_state=int(rng.integers(1 << 30)))
        legs = d.merge(lad[["player_id", "stat"]], on=["player_id", "stat"]).groupby(["player_id", "stat"]) \
            .sample(1, random_state=int(rng.integers(1 << 30)))
        side = rng.random(len(legs)) < 0.5
        p = np.where(side, legs["p_cal"], 1 - legs["p_cal"])
        y = np.where(side, legs["y"], 1 - legs["y"])
        rows.append((g, k, legs.reset_index(drop=True), side, p, int(y.all())))
    # price every sampled combo with one simulation per game
    p_ind, p_cop, ys, ks = [], [], [], []
    for g in {r[0] for r in rows}:
        rs = [r for r in rows if r[0] == g]
        d = by_game[g]
        vars_ = d.drop_duplicates(["player_id", "stat"])[["game_id", "player_id", "team", "posg", "stat"]].reset_index(drop=True)
        U = simulate(vars_, cop, 4000, seed=int(rng.integers(1 << 30)))
        vid = {k: i for i, k in enumerate(zip(vars_["player_id"], vars_["stat"]))}
        for _, k, legs, side, p, y in rs:
            idx = [vid[t] for t in zip(legs["player_id"], legs["stat"])]
            u = U[:, idx]
            h = np.where(side, u >= 1 - p, u < p).all(axis=1)
            p_cop.append(h.mean()); p_ind.append(np.prod(p)); ys.append(y); ks.append(k)
    p_ind, p_cop, ys, ks = map(np.asarray, (p_ind, p_cop, ys, ks))
    ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1))))
    out = dict(test_from=test_from, combos=int(len(ys)), hit_rate=float(ys.mean()),
               mean_p_indep=float(p_ind.mean()), mean_p_copula=float(p_cop.mean()),
               logloss_indep=ll(p_ind, ys), logloss_copula=ll(p_cop, ys), by_legs={})
    for k in sorted(set(ks)):
        m = ks == k
        out["by_legs"][int(k)] = dict(n=int(m.sum()), hit=float(ys[m].mean()), p_indep=float(p_ind[m].mean()),
                                      p_copula=float(p_cop[m].mean()), ll_indep=ll(p_ind[m], ys[m]),
                                      ll_copula=ll(p_cop[m], ys[m]))
    (REPORTS / "combo_backtest.json").write_text(json.dumps(out, indent=1))
    return out
