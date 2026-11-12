"""Final probability = market price corrected by the model, with the correction learned from settled markets.

In log-odds:

    logit(p_final) = logit(p_market) + a + a_stat + c * logit(p_market) + (b0 + b + b_stat) * d + g * d*|d|
    d = logit(p_model) - logit(p_market), clipped

`b0` is the configured prior model weight (1 - market_weight). Everything else is learned and shrunk toward
zero (ridge), so with little data this is exactly the fixed blend from the config, and a stat only moves
away from it as settled markets pile up evidence. `b_stat` is how much more or less the model earns on one
stat, `g` lets the weight change when model and market disagree a lot (usually news the model has not
seen), and `a`, `a_stat`, `c` absorb any systematic tilt in the prices themselves.

`walk_forward` scores this honestly: each week is priced by coefficients fitted on earlier weeks only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import MODELS, STATS
from .edge import expit, logit

D_CLIP = 2.5
STAT_LIST = list(STATS)
PATH = MODELS / "blend.json"


def _design(p_model, p_market, stat) -> tuple[np.ndarray, np.ndarray]:
    lm = logit(np.asarray(p_market, float))
    d = np.clip(logit(np.asarray(p_model, float)) - lm, -D_CLIP, D_CLIP)
    stat = np.asarray(stat)
    onehot = np.column_stack([(stat == s).astype(float) for s in STAT_LIST])
    X = np.column_stack([np.ones(len(lm)), lm, d, d * np.abs(d), onehot, onehot * d[:, None]])
    return lm, X


def names() -> list[str]:
    return ["a", "c", "b", "g"] + [f"a_{s}" for s in STAT_LIST] + [f"b_{s}" for s in STAT_LIST]


def fit(d: pd.DataFrame, ridge: float = 30.0, prior_w: float = 0.3) -> np.ndarray:
    """d needs p_model, p_market, stat, y. Returns the coefficient vector (see `names`), in which the global
    model weight `b` already includes the prior."""
    lm, X = _design(d["p_model"], d["p_market"], d["stat"])
    y = d["y"].to_numpy(float)
    pen = np.full(X.shape[1], ridge)
    pen[0] = ridge / 10  # the global intercept is cheap to learn and soaks up selection bias
    pen[2] = ridge / 10  # so is the global model weight
    off = lm + prior_w * X[:, 2]

    def obj(beta):
        z = off + X @ beta
        p = expit(z)
        loss = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(pen * beta ** 2)
        return loss, X.T @ (p - y) + pen * beta

    beta = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
    beta[2] += prior_w
    return beta


def apply(beta, p_model, p_market, stat) -> np.ndarray:
    lm, X = _design(p_model, p_market, stat)
    return expit(lm + X @ np.asarray(beta, float))


def model_weight(beta) -> dict[str, float]:
    """Share of the (small-disagreement) final log-odds that comes from the model, per stat."""
    n = names()
    b = dict(zip(n, beta))
    return {s: float(b["b"] + b[f"b_{s}"]) for s in STAT_LIST}


def walk_forward(d: pd.DataFrame, ridge: float = 30.0, prior_w: float = 0.3, min_weeks: int = 4) -> pd.Series:
    """Out-of-sample p_final for every row: week t uses coefficients fitted on weeks < t. Rows in the first
    `min_weeks` weeks (nothing to learn from yet) get NaN."""
    t = d["season"].to_numpy() * 100 + d["week"].to_numpy()
    out = pd.Series(np.nan, index=d.index)
    weeks = np.sort(np.unique(t))
    for w in weeks[min_weeks:]:
        tr, te = t < w, t == w
        beta = fit(d[tr], ridge, prior_w)
        out[te] = apply(beta, d.loc[te, "p_model"], d.loc[te, "p_market"], d.loc[te, "stat"])
    return out


def save(beta, meta: dict | None = None) -> None:
    PATH.write_text(json.dumps(dict(names=names(), beta=[float(x) for x in beta], meta=meta or {}), indent=1))


def load() -> np.ndarray | None:
    if not PATH.exists():
        return None
    j = json.loads(PATH.read_text())
    return np.asarray(j["beta"]) if j.get("names") == names() else None
