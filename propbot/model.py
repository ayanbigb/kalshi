"""Two-stage prop model.

Stage 1 (projection): gradient-boosted regressor per stat -> expected value `mu`.
Stage 2 (ladder):     one monotone gradient-boosted classifier per stat that answers
                      P(stat > k | mu, k, context) for *any* strike k. The strike is a feature,
                      constrained so probability can only fall as the strike rises. This prices
                      every rung of a Kalshi "N+" ladder from one model.

Stage-2 training uses stage-1 projections made by models that never saw that season
(expanding-window stacking), so nothing leaks from the future.
"""
from __future__ import annotations

import json
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import BACKTEST_SEASONS, MODELS, REPORTS, STACK_START, STATS, TRAIN_START
from .features import feature_columns

warnings.filterwarnings("ignore")
K_FEATS = ["k", "k_minus_mu", "k_log_ratio", "k_z", "mu"]
N_STAGE2_FEATS = 36
ALWAYS = ["pos_code", "games_career", "spread", "implied_total", "total_line", "is_home", "indoors", "wind",
          "inj_questionable", "inj_practice", "offense_pct_e2", "offense_pct_sd8", "vac_tgt", "vac_carry", "qb_changed"]


def rows_for(df: pd.DataFrame, stat: str, future: bool = False) -> pd.Series:
    s = STATS[stat]
    m = df["position"].isin(s["pos"]) & (df["is_future"] == future)
    if s.get("starter_qb"):
        m &= df["is_primary_qb"]
    if not future:
        m &= df[s["col"]].notna() & (df["season"] >= TRAIN_START)
    return m


def _weights(season: pd.Series) -> np.ndarray:
    return np.power(0.9, season.max() - season.to_numpy())


def fit_mean(d: pd.DataFrame, stat: str, feats: list[str]) -> lgb.LGBMRegressor:
    s = STATS[stat]
    y = d[s["col"]].clip(lower=0).to_numpy()
    small = len(d) < 8000  # QB-only stats have ~550 rows a season
    params = dict(objective=s["obj"], n_estimators=350 if small else 500, learning_rate=0.03,
                  num_leaves=15 if small else 31, min_child_samples=40 if small else 80, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, n_jobs=4, verbose=-1)
    if s["obj"] == "tweedie":
        params["tweedie_variance_power"] = 1.3
    m = lgb.LGBMRegressor(**params)
    m.fit(d[feats], y, sample_weight=_weights(d["season"]))
    return m


def stack(d: pd.DataFrame, mu: np.ndarray, feats2: list[str], ks) -> pd.DataFrame:
    """Expand rows x strikes. `ks` is either a 1-D grid applied to every row or a list of per-row arrays."""
    n = len(d)
    if n and np.ndim(ks[0]) == 0:
        reps = np.full(n, len(ks)); k = np.tile(np.asarray(ks, float), n)
    else:
        reps = np.array([len(x) for x in ks]); k = np.concatenate([np.asarray(x, float) for x in ks]) if n else np.array([])
    idx = np.repeat(np.arange(n), reps)
    X = pd.DataFrame(d[feats2].to_numpy(np.float32)[idx], columns=feats2)
    m = np.clip(np.asarray(mu, float)[idx], 0, None)
    X["k"] = k; X["k_minus_mu"] = k - m
    X["k_log_ratio"] = np.log((k + 1) / (m + 1)); X["k_z"] = (k - m) / np.sqrt(m + 1); X["mu"] = m
    X["_row"] = idx
    return X


def fit_ladder(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, feats2: list[str]) -> lgb.LGBMClassifier:
    cols = feats2 + K_FEATS
    mono = [0] * len(feats2) + [-1, -1, -1, -1, 0]
    small = len(X) < 150_000
    m = lgb.LGBMClassifier(objective="binary", n_estimators=250 if small else 400, learning_rate=0.04,
                           num_leaves=15 if small else 47, min_child_samples=150 if small else 400,
                           subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20.0,
                           monotone_constraints=mono, n_jobs=4, verbose=-1)
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


# ----------------------------------------------------------------------------- training
def train(df: pd.DataFrame, backtest: bool = True, log=print, backtest_seasons: list[int] | None = None) -> dict:
    bts = backtest_seasons or BACKTEST_SEASONS
    feats = feature_columns(df)
    hist = df[~df["is_future"]]
    last = int(hist["season"].max())
    meta, report, oos_all, calib = {}, {}, [], {}
    for stat, s in STATS.items():
        d = hist[rows_for(hist, stat)].reset_index(drop=True)
        y = d[s["col"]].to_numpy(float)
        # ---- stage 1, expanding window: mu for season S comes from a model fit on seasons < S
        mu = np.full(len(d), np.nan)
        feats2 = None
        for S in range(STACK_START, last + 1):
            tr = d["season"] < S
            te = d["season"] == S
            if not te.any():
                continue
            m1 = fit_mean(d[tr], stat, feats)
            mu[te.to_numpy()] = m1.predict(d.loc[te, feats])
            if S == bts[0]:  # choose stage-2 context features without touching test seasons
                imp = pd.Series(m1.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
                feats2 = list(dict.fromkeys([f for f in ALWAYS if f in feats] + list(imp.index)))[:N_STAGE2_FEATS]
        ok = ~np.isnan(mu)
        ds, ys, mus = d[ok].reset_index(drop=True), y[ok], mu[ok]
        X = stack(ds, mus, feats2, s["grid"])
        Y = (ys[X["_row"].to_numpy()] > X["k"].to_numpy()).astype(int)
        seas = ds["season"].to_numpy()[X["_row"].to_numpy()]
        W = np.power(0.9, last - seas)
        # ---- walk-forward backtest of the ladder model
        if backtest:
            parts = []
            for T in bts:
                tr, te = seas < T, seas == T
                if not te.any() or tr.sum() < 500:
                    continue
                m2 = fit_ladder(X[tr], Y[tr], W[tr], feats2)
                p = m2.predict_proba(X.loc[te, feats2 + K_FEATS])[:, 1]
                p = _monotone(p, X.loc[te, "_row"].to_numpy(), X.loc[te, "k"].to_numpy())
                # naive baseline: logistic on strike vs the player's rolling average only
                e6 = ds[f"{s['col']}_e6"].fillna(0).clip(lower=0).to_numpy()[X["_row"].to_numpy()]
                B = np.column_stack([X["k"], e6, X["k"] - e6, (X["k"] - e6) / np.sqrt(e6 + 1)])
                pb = LogisticRegression(max_iter=500).fit(B[tr], Y[tr]).predict_proba(B[te])[:, 1]
                r = ds.iloc[X.loc[te, "_row"].to_numpy()][["season", "week", "game_id", "player_id", "name", "position", "team"]].reset_index(drop=True)
                r["stat"] = stat; r["k"] = X.loc[te, "k"].to_numpy(); r["mu"] = X.loc[te, "mu"].to_numpy()
                r["p"] = p; r["p_base"] = pb; r["y"] = Y[te]; r["actual"] = ys[X.loc[te, "_row"].to_numpy()]
                parts.append(r)
            o = pd.concat(parts, ignore_index=True)
            # leave-one-season-out isotonic so the calibrated numbers are honest
            o["p_cal"] = o["p"]
            for T in o["season"].unique():
                a, bmask = o["season"] != T, o["season"] == T
                if a.any():
                    o.loc[bmask, "p_cal"] = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.998) \
                        .fit(o.loc[a, "p"], o.loc[a, "y"]).predict(o.loc[bmask, "p"])
            yy = o["y"].to_numpy()
            live = o["p_cal"].between(0.1, 0.9).to_numpy()  # the strikes people actually trade
            report[stat] = dict(
                model=_metrics(yy, o["p_cal"].to_numpy()), baseline=_metrics(yy, o["p_base"].to_numpy()),
                model_tradeable=_metrics(yy[live], o["p_cal"].to_numpy()[live]),
                baseline_tradeable=_metrics(yy[live], o["p_base"].to_numpy()[live]),
                ece_raw=_ece(yy, o["p"].to_numpy()), ece_cal=_ece(yy, o["p_cal"].to_numpy()),
                mae_projection=float(np.abs(o.drop_duplicates(["game_id", "player_id"])["actual"] -
                                            o.drop_duplicates(["game_id", "player_id"])["mu"]).mean()),
                by_season={int(T): _metrics(g["y"].to_numpy(), g["p_cal"].to_numpy())["logloss"] for T, g in o.groupby("season")})
            calib[stat] = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.998).fit(o["p"], o["y"])
            oos_all.append(o)
            rp = report[stat]
            log(f"  {stat:9s} n={rp['model']['n']:>7}  logloss {rp['model']['logloss']:.4f} vs naive {rp['baseline']['logloss']:.4f}"
                f"  | tradeable strikes: acc {rp['model_tradeable']['acc']:.3f} auc {rp['model_tradeable']['auc']:.3f}"
                f"  ece {rp['ece_cal']:.4f}  proj MAE {rp['mae_projection']:.2f}")
        # ---- final models on everything
        m1 = fit_mean(d, stat, feats)
        m2 = fit_ladder(X, Y, W, feats2)
        m1.booster_.save_model(str(MODELS / f"{stat}_mean.txt"))
        m2.booster_.save_model(str(MODELS / f"{stat}_ladder.txt"))
        meta[stat] = dict(feats=feats, feats2=feats2, rows=int(len(d)), last_season=last,
                          last_week=int(hist.loc[hist["season"] == last, "week"].max()))
    (MODELS / "meta.json").write_text(json.dumps(meta))
    if backtest:
        joblib.dump(calib, MODELS / "calibrators.joblib")
        pd.concat(oos_all, ignore_index=True).to_parquet(REPORTS / "oos_predictions.parquet")
        (REPORTS / "backtest.json").write_text(json.dumps(report, indent=1))
    return report


# ----------------------------------------------------------------------------- inference
class PropModel:
    def __init__(self):
        self.meta = json.loads((MODELS / "meta.json").read_text())
        self.mean = {s: lgb.Booster(model_file=str(MODELS / f"{s}_mean.txt")) for s in self.meta}
        self.ladder = {s: lgb.Booster(model_file=str(MODELS / f"{s}_ladder.txt")) for s in self.meta}
        p = MODELS / "calibrators.joblib"
        self.calib = joblib.load(p) if p.exists() else {}

    def project(self, rows: pd.DataFrame, stat: str) -> np.ndarray:
        return self.mean[stat].predict(rows[self.meta[stat]["feats"]])

    def prob(self, rows: pd.DataFrame, stat: str, strikes: list) -> pd.DataFrame:
        """rows: n feature rows; strikes: list of n arrays of floor strikes. Returns long frame
        (_row, k, mu, p) where p = P(stat > k), calibrated and monotone across each ladder."""
        f2 = self.meta[stat]["feats2"]
        mu = self.project(rows, stat)
        X = stack(rows.reset_index(drop=True), mu, f2, strikes)
        if X.empty:
            return pd.DataFrame(columns=["_row", "k", "mu", "p"])
        p = self.ladder[stat].predict(X[f2 + K_FEATS])
        if stat in self.calib:
            p = self.calib[stat].predict(p)
        p = _monotone(p, X["_row"].to_numpy(), X["k"].to_numpy())
        return pd.DataFrame({"_row": X["_row"].to_numpy(), "k": X["k"].to_numpy(), "mu": X["mu"].to_numpy(), "p": p})
