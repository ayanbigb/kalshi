"""Fees, expected value and Kelly sizing for binary contracts."""
from __future__ import annotations

import math

import numpy as np

TAKER_FEE = 0.07  # Kalshi "quadratic" fee: ceil(0.07 * contracts * price * (1 - price)) to the cent


def fee(price: float, contracts: float = 1, rate: float = TAKER_FEE) -> float:
    return math.ceil(rate * contracts * price * (1 - price) * 100 - 1e-9) / 100


def fee_per_contract(price, rate: float = TAKER_FEE):
    """Un-rounded per-contract fee (rounding happens once per order, so this is the large-order limit)."""
    return rate * price * (1 - price)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def blend(p_model, p_market, w_market: float):
    """Shrink the model toward the market in log-odds space. The market sees news the model can't."""
    return expit((1 - w_market) * logit(p_model) + w_market * logit(p_market))


def best_side(p: float, yes_ask: float, no_ask: float) -> dict | None:
    """Pick the side with the higher expected profit per contract after fees."""
    best = None
    for side, q, cost in (("yes", p, yes_ask), ("no", 1 - p, no_ask)):
        if not (cost == cost) or cost <= 0 or cost >= 1:
            continue
        c = cost + fee_per_contract(cost)
        ev = q - c
        cand = dict(side=side, win_prob=q, price=cost, cost=c, ev=ev, roi=ev / c,
                    kelly=max(0.0, (q - c) / (1 - c)))
        if best is None or cand["ev"] > best["ev"]:
            best = cand
    return best
