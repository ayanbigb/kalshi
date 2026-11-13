"""Research harness: run a model variant walk-forward and score it the same way every time.

    python -m propbot.research NAME [key=value ...]     run a variant (keys are model.OPTS), save + score it
    python -m propbot.research --compare A B ...        score saved runs side by side

Two scoreboards:
  grid    every strike in each stat's training grid, 2023-2026 (large, but mostly strikes nobody trades)
  market  only strikes Kalshi actually listed, against the real pre-kickoff quote (what the bot trades)
The market board reports the gain from blending the model into the market price, with a confidence
interval from resampling whole games, because that - not raw accuracy - is where an edge would come from.
"""
from __future__ import annotations

import ast
import json
import sys
import time

import numpy as np
import pandas as pd

from .config import DATA, REPORTS
from .edge import expit, logit


def _ll(y, p) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def quotes() -> pd.DataFrame:
    q = pd.read_parquet(DATA / "kalshi_settled.parquet")
    q = q[q["yes_bid"].between(0.005, 0.995) & q["yes_ask"].between(0.005, 0.995)].copy()
    q["p_market"] = (q["yes_bid"] + q["yes_ask"]) / 2
    q["y_mkt"] = (q["result"] == "yes").astype(int)
    return q[["ticker", "game_id", "player_id", "stat", "strike", "p_market", "yes_bid", "yes_ask", "y_mkt", "volume"]]


def on_market(oos: pd.DataFrame, q: pd.DataFrame | None = None, max_spread: float = 0.06) -> pd.DataFrame:
    q = quotes() if q is None else q
    d = q.merge(oos.rename(columns={"k": "strike"})[["game_id", "player_id", "stat", "strike", "season", "week", "p_cal", "mu", "y"]],
                on=["game_id", "player_id", "stat", "strike"], how="inner")
    # a 21c/78c quote has no information in its mid: score only quotes the bot would be allowed to trade
    return d[d["p_market"].between(0.03, 0.97) & (d["yes_ask"] - d["yes_bid"] <= max_spread)].reset_index(drop=True)


def blend_gain(d: pd.DataFrame, w: float = 0.7, n_boot: int = 500, seed: int = 0) -> dict:
    """Log-loss of market vs model vs blend on d, and the blend's gain over the market with a game-cluster CI."""
    y = d["y"].to_numpy()
    lm, lmod = _ll(y, d["p_market"]), _ll(y, d["p_cal"])
    lb = _ll(y, expit((1 - w) * logit(d["p_cal"].to_numpy()) + w * logit(d["p_market"].to_numpy())))
    ws = np.linspace(0, 1, 21)
    best = ws[int(np.argmin([_ll(y, expit((1 - a) * logit(d["p_cal"].to_numpy()) + a * logit(d["p_market"].to_numpy()))).mean() for a in ws]))]
    g = pd.DataFrame({"g": d["game_id"].to_numpy(), "d": lm - lb}).groupby("g")["d"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    s, n = g["sum"].to_numpy()[idx].sum(1), g["size"].to_numpy()[idx].sum(1)
    boots = s / n
    return dict(n=int(len(d)), games=int(len(g)), market=float(lm.mean()), model=float(lmod.mean()), blend=float(lb.mean()),
                gain=float((lm - lb).mean()), gain_lo=float(np.percentile(boots, 5)), gain_hi=float(np.percentile(boots, 95)),
                p_gain_pos=float((boots > 0).mean()), best_w=float(best))


def score(oos: pd.DataFrame, q: pd.DataFrame | None = None) -> dict:
    out = dict(grid={}, market={})
    for stat, g in oos.groupby("stat"):
        out["grid"][stat] = float(_ll(g["y"].to_numpy(), g["p_cal"]).mean())
    out["grid"]["ALL"] = float(np.mean(list(out["grid"].values())))
    if "focus" in oos.columns:  # the part of the grid that looks like listed markets
        f = oos[oos["focus"]]
        out["grid"]["FOCUS"] = float(np.mean([_ll(g["y"].to_numpy(), g["p_cal"]).mean() for _, g in f.groupby("stat")]))
    d = on_market(oos, q)
    for stat, g in d.groupby("stat"):
        out["market"][stat] = blend_gain(g)
    out["market"]["ALL"] = blend_gain(d)
    for S, g in d.groupby("season"):
        out["market"][f"season_{S}"] = blend_gain(g)
    return out


def show(scores: dict[str, dict]) -> None:
    names = list(scores)
    stats = [k for k in scores[names[0]]["grid"] if k != "FOCUS"]
    print("\nGRID log-loss (lower is better)")
    print(pd.DataFrame({n: scores[n]["grid"] for n in names}).reindex(stats + ["FOCUS"]).round(4).to_string())
    keys = list(scores[names[0]]["market"])
    print("\nMARKET strikes: model-alone log-loss")
    print(pd.DataFrame({n: {k: scores[n]["market"][k]["model"] for k in keys if k in scores[n]["market"]} for n in names})
          .assign(market=pd.Series({k: scores[names[0]]["market"][k]["market"] for k in keys}),
                  n=pd.Series({k: scores[names[0]]["market"][k]["n"] for k in keys})).round(4).to_string())
    print("\nMARKET strikes: gain of the 70/30 market/model blend over the market alone, x1000 (higher is better; [5%, 95%])")
    rows = {}
    for n in names:
        rows[n] = {k: f"{1000 * v['gain']:+.2f} [{1000 * v['gain_lo']:+.2f},{1000 * v['gain_hi']:+.2f}] w*={v['best_w']:.2f}"
                   for k, v in scores[n]["market"].items()}
    print(pd.DataFrame(rows).reindex(keys).to_string())


def run(name: str, **opts) -> dict:
    from . import model
    from .data import load_all
    from .features import build
    frame_path = DATA / f"{opts.pop('frame', 'frame')}.parquet"
    model.OPTS.update(opts)
    t0 = time.time()
    if frame_path.exists():
        df = pd.read_parquet(frame_path)
    else:
        df = build(load_all(), include_future=False)
        df.to_parquet(frame_path)
    res = model.train(df, backtest=True, final=False)
    oos = res["oos"]
    oos.to_parquet(REPORTS / f"exp_{name}.parquet")
    sc = score(oos)
    sc["opts"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in model.OPTS.items()}
    sc["minutes"] = round((time.time() - t0) / 60, 1)
    (REPORTS / f"exp_{name}.json").write_text(json.dumps(sc, indent=1))
    return sc


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--compare":
        q = quotes()
        show({n: score(pd.read_parquet(REPORTS / f"exp_{n}.parquet"), q) for n in argv[1:]})
        return
    name, opts = argv[0], {}
    for kv in argv[1:]:
        k, v = kv.split("=", 1)
        try:
            opts[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            opts[k] = v
    sc = run(name, **opts)
    show({name: sc})
    print(f"\n{sc['minutes']} min")


if __name__ == "__main__":
    main(sys.argv[1:])
