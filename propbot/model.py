"""Three-stage prop model.

Stage 1  (projection)  gradient-boosted regressor per stat -> expected value `mu`.
                       Also fitted for two helper targets (targets, snap share) that no market
                       trades but that tell the ladder model how a player gets his numbers.
Stage 1b (spread)      regressor on |actual - mu| -> `sd`, how wide this player's outcomes run.
Stage 2  (ladder)      one monotone gradient-boosted classifier per stat that answers
                       P(stat > k | mu, sd, k, projections of related stats, how often the player
                       cleared k recently, context) for *any* strike k. The strike is a feature,
                       constrained so probability can only fall as the strike rises. This prices
                       every rung of a Kalshi "N+" ladder from one model.

Everything stage 2 sees was produced by models that never saw that game (expanding-window
stacking), so nothing leaks from the future. Fits run in parallel worker processes.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import BACKTEST_SEASONS, DATA, MODELS, REPORTS, STACK_START, STATS, TRAIN_START
from .features import HIST, feature_columns, history_columns

warnings.filterwarnings("ignore")

# helper targets: projected like a stat, used only as stage-2 context
AUX = {
    "targets": dict(col="targets", pos=["WR", "TE", "RB", "FB"], obj="poisson"),
    "snap_pct": dict(col="offense_pct", pos=["QB", "RB", "FB", "WR", "TE"], obj="regression"),
}
TARGETS = {**STATS, **AUX}
# projections of these other targets are handed to each stat's ladder model
RELATED = {
    "pass_yds": ["pass_att", "pass_comp", "pass_tds"],
    "pass_tds": ["pass_yds", "pass_att"],
    "pass_att": ["pass_comp", "pass_yds"],
    "pass_comp": ["pass_att", "pass_yds"],
    "pass_int": ["pass_att", "pass_yds"],
    "rush_yds": ["rush_att", "snap_pct", "td"],
    "rush_att": ["rush_yds", "snap_pct"],
    "rec": ["rec_yds", "targets", "snap_pct"],
    "rec_yds": ["rec", "targets", "snap_pct"],
    "rr_yds": ["rush_yds", "rec_yds", "rush_att", "rec", "targets", "snap_pct"],
    "td": ["rr_yds", "rush_yds", "rec_yds", "rush_att", "rec", "targets", "snap_pct"],
}
# Kalshi only lists players with a real role, at strikes near their projection, while the history is mostly
# bit-part players and strikes nobody would quote. FOCUS = (min projection, lowest k_z, highest k_z) fences
# off the region that looks like listed markets (a little wider than what has been listed so far); with
# OPTS["focus"] the ladder model and its calibration learn from that region only.
FOCUS = {
    "pass_yds": (150, -9, 14), "pass_tds": (0.7, -1.2, 2.2), "pass_att": (22, -2.2, 2.6), "pass_comp": (13, -2.2, 2.6),
    "pass_int": (0.3, -0.6, 2.2), "rush_yds": (3, -6, 18), "rush_att": (2, -2.2, 4.0), "rec": (0.7, -1.6, 3.5),
    "rec_yds": (6, -6.5, 19), "rr_yds": (15, -4, 16), "td": (0.03, -0.5, 2.0),
}
K_FEATS = ["k", "k_minus_mu", "k_log_ratio", "k_z", "k_zs", "emp_long", "emp_short", "emp_n", "mu", "sd"]
K_MONO = [-1, -1, -1, -1, -1, 1, 1, 0, 0, 0]
ALWAYS = ["pos_code", "games_career", "spread", "implied_total", "total_line", "is_home", "indoors", "wind",
          "inj_questionable", "inj_practice", "offense_pct_e2", "offense_pct_sd8", "vac_tgt", "vac_carry", "qb_changed"]

# knobs the research harness flips; the defaults are what `train` ships
OPTS = dict(
    n_stage2_feats=36,      # context features given to the ladder model
    cross=True,             # hand projections of related stats to the ladder model
    disp=True,              # stage 1b spread model
    emp=True,               # per-strike historical hit rates
    use_sd=True,            # let the ladder model see the stage 1b spread
    bags=1,                 # ladder models averaged (different seeds)
    focus=False,            # train + calibrate the ladder only on market-like rows (see FOCUS)
    split_weeks=(1,),       # a backtest season is cut into blocks starting at these weeks; each block is
                            # predicted by models fitted on everything before it
    drop_feats=None,        # regex of stage-1 features to leave out (ablations)
    jobs=4,
)


def rows_for(df: pd.DataFrame, stat: str, future: bool = False) -> pd.Series:
    s = TARGETS[stat]
    m = df["position"].isin(s["pos"]) & (df["is_future"] == future)
    if s.get("starter_qb"):
        m &= df["is_primary_qb"]
    if not future:
        m &= df[s["col"]].notna() & (df["season"] >= TRAIN_START)
    return m


def model_features(df: pd.DataFrame) -> list[str]:
    feats = feature_columns(df)
    if OPTS["drop_feats"]:
        rx = re.compile(OPTS["drop_feats"])
        feats = [f for f in feats if not rx.search(f)]
    return feats


# ----------------------------------------------------------------------------- fitting primitives
def fit_mean(X, y, w, spec: dict, feats: list[str]) -> lgb.LGBMRegressor:
    small = len(y) < 8000  # QB-only stats have ~550 rows a season
    params = dict(objective=spec["obj"], n_estimators=350 if small else 500, learning_rate=0.03,
                  num_leaves=15 if small else 31, min_child_samples=40 if small else 80, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, max_bin=127, n_jobs=1, verbose=-1)
    if spec["obj"] == "tweedie":
        params["tweedie_variance_power"] = 1.3
    m = lgb.LGBMRegressor(**params)
    m.fit(X, np.clip(y, 0, None), sample_weight=w, feature_name=list(feats))
    return m


def fit_disp(X, absdev, w, feats: list[str]) -> lgb.LGBMRegressor:
    """Expected absolute miss of the projection: a direct read on how wide this player's outcomes are."""
    small = len(absdev) < 8000
    m = lgb.LGBMRegressor(objective="regression", n_estimators=200 if small else 300, learning_rate=0.03,
                          num_leaves=7 if small else 15, min_child_samples=60 if small else 150, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, reg_lambda=20.0, max_bin=127, n_jobs=1, verbose=-1)
    m.fit(X, absdev, sample_weight=w, feature_name=list(feats) + ["mu"])
    return m


def stack(ctx: pd.DataFrame, mu, sd, H, ks) -> pd.DataFrame:
    """Expand rows x strikes. `ctx` holds the per-row context columns, `H` the previous games' values of the
    stat (rows x HIST, NaN = no game), `ks` either a 1-D grid applied to every row or a list of per-row arrays."""
    n = len(ctx)
    if n and np.ndim(ks[0]) == 0:
        reps = np.full(n, len(ks)); k = np.tile(np.asarray(ks, float), n)
    else:
        reps = np.array([len(x) for x in ks]); k = np.concatenate([np.asarray(x, float) for x in ks]) if n else np.array([])
    idx = np.repeat(np.arange(n), reps)
    X = pd.DataFrame(ctx.to_numpy(np.float32)[idx], columns=list(ctx.columns))
    m = np.clip(np.asarray(mu, float)[idx], 0, None)
    s = np.asarray(sd, float)[idx]
    X["k"] = k; X["k_minus_mu"] = k - m
    X["k_log_ratio"] = np.log((k + 1) / (m + 1)); X["k_z"] = (k - m) / np.sqrt(m + 1)
    X["k_zs"] = (k - m) / np.clip(s, 0.25, None)
    h = np.asarray(H, np.float32)[idx] if n else np.zeros((0, HIST), np.float32)
    seen = ~np.isnan(h)
    over = (h > k[:, None]) & seen
    with np.errstate(invalid="ignore", divide="ignore"):
        X["emp_long"] = over.sum(1) / np.where(seen.sum(1) > 0, seen.sum(1), np.nan)
        X["emp_short"] = over[:, :5].sum(1) / np.where(seen[:, :5].sum(1) > 0, seen[:, :5].sum(1), np.nan)
    X["emp_n"] = seen.sum(1).astype(np.float32)
    X["mu"] = m; X["sd"] = s
    if not OPTS["emp"]:
        X[["emp_long", "emp_short", "emp_n"]] = np.nan
    if not OPTS["use_sd"]:
        X[["k_zs", "sd"]] = np.nan
    X["_row"] = idx
    return X


def fit_ladder(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, ctx_cols: list[str], seed: int = 0) -> lgb.LGBMClassifier:
    cols = ctx_cols + K_FEATS
    mono = [0] * len(ctx_cols) + K_MONO
    small = len(X) < 150_000
    m = lgb.LGBMClassifier(objective="binary", n_estimators=250 if small else 400, learning_rate=0.04,
                           num_leaves=15 if small else 47, min_child_samples=150 if small else 400,
                           subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20.0,
                           monotone_constraints=mono, max_bin=127, n_jobs=1, verbose=-1, random_state=seed)
    m.fit(X[cols], y, sample_weight=w)
    return m


def _monotone(p: np.ndarray, row: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Enforce P non-increasing in k within each row (cumulative min along sorted strikes)."""
    o = pd.DataFrame({"p": p, "r": row, "k": k})
    o["p"] = o.sort_values(["r", "k"]).groupby("r")["p"].cummin()
    return o["p"].to_numpy()


def _metrics(y, p) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return dict(n=int(len(y)), logloss=float(log_loss(y, p)), brier=float(brier_score_loss(y, p)),
                auc=float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
                acc=float(((p > 0.5) == y).mean()))


def _ece(y, p, bins=10) -> float:
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))


# ----------------------------------------------------------------------------- parallel workers
_G: dict = {}  # filled by the parent right before forking; workers read it (copy-on-write, nothing is pickled)


def _t(season, week) -> np.ndarray:
    return np.asarray(season) * 100 + np.asarray(week)


def _blocks(seasons, last_t: int) -> list[tuple[int, int]]:
    """[start, end) time blocks: each listed season cut at OPTS['split_weeks']."""
    out = []
    for S in seasons:
        cuts = [S * 100 + w for w in OPTS["split_weeks"]] + [(S + 1) * 100]
        out += [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if a <= last_t]
    return out


def _mean_task(a):
    name, blk = a
    g = _G; rows = g["rows"][name]; t = g["t"][rows]; y = g["y"][name]
    tr = t < blk[0] if blk else np.ones(len(rows), bool)
    te = (t >= blk[0]) & (t < blk[1]) if blk else np.zeros(len(rows), bool)
    if blk and (not te.any() or tr.sum() < 300):
        return name, blk, None, None, None
    w = np.power(0.9, g["season"][rows][tr].max() - g["season"][rows][tr])
    m = fit_mean(g["X"][rows[tr]], y[tr], w, TARGETS[name], g["feats"])
    pred = m.predict(g["X"][rows[te]]) if blk else None
    imp = m.booster_.feature_importance("gain") if (blk and blk[0] == g["imp_block"]) else None
    return name, blk, pred, imp, (m.booster_.model_to_string() if not blk else None)


def _disp_task(a):
    name, blk = a
    g = _G; rows = g["rows"][name]; t = g["t"][rows]; y = g["y"][name]; mu = g["mu"][name]
    ok = ~np.isnan(mu)
    tr = ok & (t < blk[0]) if blk else ok
    te = ok & (t >= blk[0]) & (t < blk[1]) if blk else np.zeros(len(rows), bool)
    if tr.sum() < 300 or (blk and not te.any()):
        return name, blk, None, None
    Xm = lambda mask: np.column_stack([g["X"][rows[mask]], mu[mask].astype(np.float32)])
    w = np.power(0.9, g["season"][rows][tr].max() - g["season"][rows][tr])
    m = fit_disp(Xm(tr), np.abs(y[tr] - mu[tr]), w, g["feats"])
    return name, blk, (m.predict(Xm(te)) if blk else None), (m.booster_.model_to_string() if not blk else None)


def _ladder_data(stat: str):
    """Stacked (row x strike) design for one stat from the out-of-fold stage-1 outputs in _G."""
    g = _G; rows = g["rows"][stat]; mu = g["mu"][stat]
    ok = ~np.isnan(mu)
    r = rows[ok]
    f2 = g["feats2"][stat]
    ctx = pd.DataFrame(g["X"][r][:, [g["fidx"][f] for f in f2]], columns=f2)
    for o in g["cross"][stat]:
        full = np.full(len(g["t"]), np.nan); full[g["rows"][o]] = g["mu"][o]
        ctx[f"mu_{o}"] = full[r]
    sd = g["sd"][stat][ok] if stat in g["sd"] else np.full(len(r), np.nan)
    X = stack(ctx, mu[ok], sd, g["H"][stat][r], STATS[stat]["grid"])
    ri = X["_row"].to_numpy()
    Y = (g["y"][stat][ok][ri] > X["k"].to_numpy()).astype(int)
    return X, Y, r, ri, list(ctx.columns)


def _ladder_task(a):
    stat, blk = a
    g = _G
    X, Y, r, ri, cc = _ladder_data(stat)
    t = g["t"][r][ri]; seas = g["season"][r][ri]
    lo_mu, lo_z, hi_z = FOCUS[stat]
    focus = ((X["mu"] >= lo_mu) & X["k_z"].between(lo_z, hi_z)).to_numpy()
    tr = t < blk[0] if blk else np.ones(len(t), bool)
    te = (t >= blk[0]) & (t < blk[1]) if blk else np.zeros(len(t), bool)
    if OPTS["focus"]:
        tr &= focus
    if blk and (not te.any() or tr.sum() < 500):
        return stat, blk, None, None
    W = np.power(0.9, seas[tr].max() - seas[tr])
    ms = [fit_ladder(X[tr], Y[tr], W, cc, seed) for seed in range(OPTS["bags"])]
    if not blk:
        return stat, blk, None, [m.booster_.model_to_string() for m in ms]
    Xt = X.loc[te, cc + K_FEATS]
    p = np.mean([m.predict_proba(Xt)[:, 1] for m in ms], axis=0)
    p = _monotone(p, ri[te], X.loc[te, "k"].to_numpy())
    # naive baseline: logistic on strike vs the player's rolling average only
    e6 = np.nan_to_num(g["e6"][stat][r], nan=0.0).clip(min=0)[ri]
    k = X["k"].to_numpy()
    B = np.column_stack([k, e6, k - e6, (k - e6) / np.sqrt(e6 + 1)])
    pb = LogisticRegression(max_iter=500).fit(B[tr], Y[tr]).predict_proba(B[te])[:, 1]
    out = g["ids"].iloc[r[ri[te]]].reset_index(drop=True)
    out["stat"] = stat; out["k"] = k[te]; out["mu"] = X.loc[te, "mu"].to_numpy(); out["sd"] = X.loc[te, "sd"].to_numpy()
    out["p"] = p; out["p_base"] = pb; out["y"] = Y[te]; out["actual"] = g["y"][stat][~np.isnan(g["mu"][stat])][ri[te]]
    out["focus"] = focus[te]
    return stat, blk, out, None


def _pmap(fn, tasks, log=None, label=""):
    """Run tasks in forked workers (largest first so the pool stays busy)."""
    t0 = time.time()
    if OPTS["jobs"] <= 1 or len(tasks) <= 1:
        res = [fn(a) for a in tasks]
    else:
        with mp.get_context("fork").Pool(min(OPTS["jobs"], len(tasks))) as pool:
            res = list(pool.imap_unordered(fn, tasks, chunksize=1))
    if log:
        log(f"  {label}: {len(tasks)} fits in {time.time() - t0:.0f}s")
    return res


# ----------------------------------------------------------------------------- training
def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feats = model_features(df)
    hist = df[~df["is_future"]].reset_index(drop=True)
    g = _G
    g.clear()
    g["feats"] = feats; g["fidx"] = {f: i for i, f in enumerate(feats)}
    g["X"] = hist[feats].to_numpy(np.float32)
    g["season"] = hist["season"].to_numpy(int); g["t"] = _t(hist["season"], hist["week"])
    g["rows"] = {n: np.flatnonzero(rows_for(hist, n).to_numpy()) for n in TARGETS}
    g["y"] = {n: hist[TARGETS[n]["col"]].to_numpy(float)[g["rows"][n]] for n in TARGETS}
    g["H"] = {s: hist[history_columns(hist, STATS[s]["col"])].to_numpy(np.float32) for s in STATS}
    g["e6"] = {s: hist[f"{STATS[s]['col']}_e6"].to_numpy(float) for s in STATS}
    g["ids"] = hist[["season", "week", "game_id", "player_id", "name", "position", "team"]]
    g["cross"] = {s: (RELATED[s] if OPTS["cross"] else []) for s in STATS}
    return hist, feats


def _stage1(hist: pd.DataFrame, final: bool, log) -> dict:
    """Out-of-fold projections + spreads for every target; cached on disk for the research loop."""
    g = _G
    last = int(hist["season"].max()); last_t = int(g["t"].max())
    blocks = _blocks(range(STACK_START, last + 1), last_t)
    g["imp_block"] = BACKTEST_SEASONS[0] * 100 + OPTS["split_weeks"][0]
    key = hashlib.md5(json.dumps([g["feats"], blocks, len(hist), last_t, OPTS["disp"]]).encode()).hexdigest()[:12]
    cache = DATA / "cache" / f"stage1_{key}.joblib"
    out = dict(models={}, disp_models={})
    if cache.exists():
        c = joblib.load(cache)
        g["mu"], g["sd"], g["imp"] = c["mu"], c["sd"], c["imp"]
        log(f"  stage 1: reused cached out-of-fold projections ({key})")
    else:
        g["mu"] = {n: np.full(len(g["rows"][n]), np.nan) for n in TARGETS}; g["imp"] = {}
        for name, blk, pred, imp, _ in _pmap(_mean_task, [(n, b) for b in blocks for n in TARGETS], log, "stage 1 projections"):
            if pred is None:
                continue
            t = g["t"][g["rows"][name]]
            g["mu"][name][(t >= blk[0]) & (t < blk[1])] = pred
            if imp is not None:
                g["imp"][name] = imp
        g["sd"] = {}
        if OPTS["disp"]:
            g["sd"] = {n: np.full(len(g["rows"][n]), np.nan) for n in STATS}
            for name, blk, pred, _ in _pmap(_disp_task, [(n, b) for b in blocks[1:] for n in STATS], log, "stage 1b spreads"):
                if pred is None:
                    continue
                t = g["t"][g["rows"][name]]
                g["sd"][name][~np.isnan(g["mu"][name]) & (t >= blk[0]) & (t < blk[1])] = pred
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(mu=g["mu"], sd=g["sd"], imp=g["imp"]), cache)
    g["feats2"] = {}
    for s in STATS:
        imp = pd.Series(g["imp"][s], index=g["feats"]).sort_values(ascending=False)
        g["feats2"][s] = list(dict.fromkeys([f for f in ALWAYS if f in g["fidx"]] + list(imp.index)))[:OPTS["n_stage2_feats"]]
    if final:
        for name, _, _, _, ms in _pmap(_mean_task, [(n, None) for n in TARGETS], log, "final projections"):
            out["models"][name] = ms
        if OPTS["disp"]:
            for name, _, _, ms in _pmap(_disp_task, [(n, None) for n in STATS], log, "final spreads"):
                out["disp_models"][name] = ms
    return out


def train(df: pd.DataFrame, backtest: bool = True, final: bool = True, log=print, stats: list[str] | None = None) -> dict:
    hist, feats = _prepare(df)
    g = _G
    last = int(hist["season"].max()); last_t = int(g["t"].max())
    s1 = _stage1(hist, final, log)
    stats = stats or list(STATS)
    report, oos_all, calib = {}, [], {}
    if backtest:
        blocks = _blocks(BACKTEST_SEASONS, last_t)
        parts: dict[str, list] = {s: [] for s in stats}
        for stat, _, o, _ in _pmap(_ladder_task, [(s, b) for b in blocks for s in stats], log, "ladder backtest"):
            if o is not None:
                parts[stat].append(o)
        for stat in stats:
            o = pd.concat(parts[stat], ignore_index=True)
            cal_rows = o["focus"] if OPTS["focus"] else pd.Series(True, index=o.index)
            # leave-one-season-out isotonic so the calibrated numbers are honest
            o["p_cal"] = o["p"]
            for T in o["season"].unique():
                a, bmask = (o["season"] != T) & cal_rows, o["season"] == T
                if a.any():
                    o.loc[bmask, "p_cal"] = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.998) \
                        .fit(o.loc[a, "p"], o.loc[a, "y"]).predict(o.loc[bmask, "p"])
            yy = o["y"].to_numpy()
            live = o["p_cal"].between(0.1, 0.9).to_numpy()  # the strikes people actually trade
            one = o.drop_duplicates(["game_id", "player_id"])
            report[stat] = dict(
                model=_metrics(yy, o["p_cal"].to_numpy()), baseline=_metrics(yy, o["p_base"].to_numpy()),
                model_tradeable=_metrics(yy[live], o["p_cal"].to_numpy()[live]),
                baseline_tradeable=_metrics(yy[live], o["p_base"].to_numpy()[live]),
                ece_raw=_ece(yy, o["p"].to_numpy()), ece_cal=_ece(yy, o["p_cal"].to_numpy()),
                mae_projection=float(np.abs(one["actual"] - one["mu"]).mean()),
                by_season={int(T): _metrics(gp["y"].to_numpy(), gp["p_cal"].to_numpy())["logloss"] for T, gp in o.groupby("season")})
            calib[stat] = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.998) \
                .fit(o.loc[cal_rows, "p"], o.loc[cal_rows, "y"])
            oos_all.append(o)
            rp = report[stat]
            log(f"  {stat:9s} n={rp['model']['n']:>7}  logloss {rp['model']['logloss']:.4f} vs naive {rp['baseline']['logloss']:.4f}"
                f"  | tradeable strikes: acc {rp['model_tradeable']['acc']:.3f} auc {rp['model_tradeable']['auc']:.3f}"
                f"  ece {rp['ece_cal']:.4f}  proj MAE {rp['mae_projection']:.2f}")
    oos = pd.concat(oos_all, ignore_index=True) if oos_all else pd.DataFrame()
    if final:
        ladders = {stat: ms for stat, _, _, ms in _pmap(_ladder_task, [(s, None) for s in stats], log, "final ladders")}
        for f in MODELS.glob("*.txt"):
            f.unlink()
        meta = {}
        for name, ms in s1["models"].items():
            (MODELS / f"{name}_mean.txt").write_text(ms)
        for name, ms in s1["disp_models"].items():
            (MODELS / f"{name}_disp.txt").write_text(ms)
        for stat, mss in ladders.items():
            for i, ms in enumerate(mss):
                (MODELS / f"{stat}_ladder{i}.txt").write_text(ms)
            meta[stat] = dict(feats=feats, feats2=g["feats2"][stat], cross=g["cross"][stat], bags=len(mss),
                              disp=stat in s1["disp_models"], rows=int(len(g["rows"][stat])), last_season=last,
                              last_week=int(hist.loc[hist["season"] == last, "week"].max()))
        (MODELS / "meta.json").write_text(json.dumps(dict(stats=meta, aux=list(AUX), feats=feats)))
        if backtest:
            joblib.dump(calib, MODELS / "calibrators.joblib")
    if backtest and final:
        oos.to_parquet(REPORTS / "oos_predictions.parquet")
        (REPORTS / "backtest.json").write_text(json.dumps(report, indent=1))
    return dict(report=report, oos=oos)


# ----------------------------------------------------------------------------- inference
class PropModel:
    def __init__(self):
        meta = json.loads((MODELS / "meta.json").read_text())
        self.meta, self.feats = meta["stats"], meta["feats"]
        load = lambda f: lgb.Booster(model_file=str(MODELS / f))
        self.mean = {s: load(f"{s}_mean.txt") for s in list(self.meta) + meta["aux"]}
        self.disp = {s: load(f"{s}_disp.txt") for s, m in self.meta.items() if m["disp"]}
        self.ladder = {s: [load(f"{s}_ladder{i}.txt") for i in range(m["bags"])] for s, m in self.meta.items()}
        p = MODELS / "calibrators.joblib"
        self.calib = joblib.load(p) if p.exists() else {}

    def project(self, rows: pd.DataFrame, stat: str) -> np.ndarray:
        return self.mean[stat].predict(rows[self.feats].to_numpy(np.float32))

    def _eligible(self, rows: pd.DataFrame, stat: str) -> np.ndarray:
        s = TARGETS[stat]
        m = rows["position"].isin(s["pos"])
        if s.get("starter_qb"):
            m &= rows["is_primary_qb"].astype(bool)
        return m.to_numpy()

    def prob(self, rows: pd.DataFrame, stat: str, strikes: list) -> pd.DataFrame:
        """rows: n feature rows; strikes: list of n arrays of floor strikes. Returns long frame
        (_row, k, mu, sd, p) where p = P(stat > k), calibrated and monotone across each ladder."""
        m = self.meta[stat]
        rows = rows.reset_index(drop=True)
        if not len(rows):
            return pd.DataFrame(columns=["_row", "k", "mu", "sd", "p"])
        Xf = rows[self.feats].to_numpy(np.float32)
        mu = self.mean[stat].predict(Xf)
        ctx = rows[m["feats2"]].astype(np.float32).copy()
        for o in m["cross"]:
            ctx[f"mu_{o}"] = np.where(self._eligible(rows, o), self.mean[o].predict(Xf), np.nan)
        sd = self.disp[stat].predict(np.column_stack([Xf, mu.astype(np.float32)])) if stat in self.disp else np.full(len(rows), np.nan)
        H = rows[history_columns(rows, STATS[stat]["col"])].to_numpy(np.float32)
        X = stack(ctx, mu, sd, H, strikes)
        if X.empty:
            return pd.DataFrame(columns=["_row", "k", "mu", "sd", "p"])
        cols = list(ctx.columns) + K_FEATS
        p = np.mean([b.predict(X[cols]) for b in self.ladder[stat]], axis=0)
        if stat in self.calib:
            p = self.calib[stat].predict(p)
        p = _monotone(p, X["_row"].to_numpy(), X["k"].to_numpy())
        return pd.DataFrame({"_row": X["_row"].to_numpy(), "k": X["k"].to_numpy(), "mu": X["mu"].to_numpy(),
                             "sd": X["sd"].to_numpy(), "p": p})
