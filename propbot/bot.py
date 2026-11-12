"""Scan -> price -> gate -> size -> (paper|live) execute, plus settlement and the market backtest."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import odds, risk
from .config import REPORTS, DATA, STATS, env, load_config
from .data import current_season, load_all
from .edge import best_side, blend, expit, fee, fee_per_contract, logit
from .features import build
from .kalshi import Kalshi, KalshiError
from .ledger import Ledger
from .markets import attach_players, norm_name, parse
from .model import PropModel

ET = "America/New_York"


def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET).tz_localize(None)


def price_markets(mk: pd.DataFrame, frame: pd.DataFrame, model: PropModel, book: pd.DataFrame | None = None,
                  book_weight: float = 0.5) -> pd.DataFrame:
    """Attach p_model (and projection mu) to every market row that maps to a feature row."""
    mk = mk.dropna(subset=["game_id", "player_id"]).copy()
    out = []
    for stat, g in mk.groupby("stat"):
        s = STATS[stat]
        rows = frame[frame["position"].isin(s["pos"])]
        if s.get("starter_qb"):
            rows = rows[rows["is_primary_qb"]]
        g = g.merge(rows[["game_id", "player_id"]].drop_duplicates(), on=["game_id", "player_id"], how="inner")
        if g.empty:
            continue
        ladders = g.groupby(["game_id", "player_id"])["strike"].apply(lambda x: np.sort(x.unique())).reset_index()
        feat = ladders[["game_id", "player_id"]].merge(rows, on=["game_id", "player_id"], how="left").reset_index(drop=True)
        strikes = list(ladders["strike"])
        shift = np.zeros(len(feat))
        if book is not None and len(book):
            b = book[book["stat"] == stat].set_index("nn")
            nn = feat["name"].map(norm_name)
            has = nn.isin(b.index).to_numpy()
            if has.any():
                sub = feat[has].reset_index(drop=True)
                lines = [np.array([b.loc[n, "line"]]) for n in nn[has]]
                at = model.prob(sub, stat, lines)["p"].to_numpy()
                pb = np.array([b.loc[n, "p_book"] for n in nn[has]])
                shift[has] = [odds.logit_shift(a, p, book_weight) for a, p in zip(at, pb)]
        pr = model.prob(feat, stat, strikes)
        pr["p"] = expit(logit(pr["p"].to_numpy()) + shift[pr["_row"].to_numpy()])
        pr["game_id"] = feat["game_id"].to_numpy()[pr["_row"]]
        pr["player_id"] = feat["player_id"].to_numpy()[pr["_row"]]
        pr["book_shift"] = shift[pr["_row"].to_numpy()]
        keep = ["name", "position", "kickoff", "games_career", "inj_questionable"]
        pr = pr.merge(feat[["game_id", "player_id"] + keep], on=["game_id", "player_id"])
        out.append(g.merge(pr.rename(columns={"k": "strike", "p": "p_model"}).drop(columns="_row"),
                           on=["game_id", "player_id", "strike"], how="inner"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def add_edges(px: pd.DataFrame, cfg: dict, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if px.empty:
        return px
    now = now or _now_et()
    r, w = cfg["risk"], cfg["model"]["market_weight"]
    two_sided = px["yes_bid"].between(0.005, 0.995) & px["yes_ask"].between(0.005, 0.995)
    px["p_market"] = np.where(two_sided, (px["yes_bid"] + px["yes_ask"]) / 2, np.nan)
    px["spread"] = np.where(two_sided, px["yes_ask"] - px["yes_bid"], np.nan)
    px["p_final"] = np.where(two_sided, blend(px["p_model"], px["p_market"].fillna(0.5), w), np.nan)
    px["minutes_to_kickoff"] = (pd.to_datetime(px["kickoff"]) - now).dt.total_seconds() / 60
    recs = []
    for c in px.to_dict("records"):
        b = best_side(c["p_final"], c["yes_ask"], c["no_ask"]) if c["p_final"] == c["p_final"] else None
        if b is None:
            b = dict(side=None, win_prob=np.nan, price=np.nan, cost=np.nan, ev=-1.0, roi=-1.0, kelly=0.0)
        c.update(b)
        c["size"] = c["yes_ask_size"] if c["side"] == "yes" else c["no_ask_size"]
        c["fails"] = ",".join(risk.gate_row(c, r)) if c["side"] else "no_quote"
        recs.append(c)
    return pd.DataFrame(recs)


def scan(cfg: dict | None = None, client: Kalshi | None = None, refresh: bool = False, log=print) -> pd.DataFrame:
    cfg = cfg or load_config()
    client = client or Kalshi("prod")  # quotes always come from production
    mk = pd.concat([parse(client.markets(s, "open")) for s in cfg["kalshi"]["series"]], ignore_index=True)
    log(f"  {len(mk)} open prop markets across {mk['event'].nunique() if len(mk) else 0} events")
    if mk.empty:
        return mk
    raw = load_all(refresh=refresh)
    frame = build(raw)
    fut = frame[frame["is_future"]]
    mk = attach_players(mk, fut)
    # the QB Kalshi lists passing props for is the expected starter
    q = mk[(mk["stat"] == "pass_yds") & mk["player_id"].notna()].groupby("team")["player_id"].agg(lambda s: s.mode().iloc[0])
    cur = fut[fut["is_primary_qb"]].drop_duplicates("team").set_index("team")["player_id"]
    if any(cur.get(t) != p for t, p in q.items()):
        frame = build(raw, qb_override=q.to_dict())
        fut = frame[frame["is_future"]]
    nxt = fut.groupby("team")["kickoff"].transform("min")
    fut = fut[fut["kickoff"] == nxt]  # only a team's next game has up-to-date features
    log(f"  matched {mk['player_id'].notna().mean():.0%} of markets to players; "
        f"{fut['game_id'].nunique()} upcoming games in the feature frame")
    book = odds.fetch_props()
    if len(book):
        log(f"  sportsbook anchor: {len(book)} player lines")
    px = price_markets(mk, fut, PropModel(), book, cfg["model"]["book_weight"])
    return add_edges(px, cfg)


def _state(ledger: Ledger, mode: str, cfg: dict, client: Kalshi | None) -> dict:
    st = ledger.summary(mode, cfg["risk"]["bankroll"])
    if mode == "live":
        cash = client.balance()
        st.update(cash=cash, bankroll=cash + st["at_risk"])
    return st


def run_once(live: bool = False, refresh: bool = False, log=print) -> pd.DataFrame:
    cfg = load_config()
    mode = "live" if live else "paper"
    ledger = Ledger()
    trader = None
    if live:
        if env("KALSHI_LIVE_TRADING") != "1":
            raise SystemExit("Live trading is off. Set KALSHI_LIVE_TRADING=1 in .env to enable it.")
        trader = Kalshi(cfg["kalshi"]["env"])
        if not trader.can_trade:
            raise SystemExit("Missing KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH in .env")
    state = _state(ledger, mode, cfg, trader)
    tripped = risk.breakers(state, cfg["risk"])
    if tripped:
        log("  HALTED: " + "; ".join(tripped))
        return pd.DataFrame()
    px = scan(cfg, refresh=refresh, log=log)
    if px.empty:
        return px
    px.to_parquet(REPORTS / "last_scan.parquet")
    ok = px[px["fails"] == ""]
    log(f"  {len(px)} priced markets, {len(ok)} pass gates 1-4")
    orders = risk.size_orders(ok, ledger.trades(mode, "open"), state, cfg["risk"])
    for o in orders.to_dict("records"):
        n, price, fees, oid = o["contracts"], o["price"], o["fees"], None
        if live:
            try:
                resp = trader.buy(o["ticker"], o["side"], n, price)
            except KalshiError as e:
                log(f"  order rejected {o['ticker']}: {e}")
                continue
            n = int(float(resp.get("fill_count", 0)))
            oid = resp.get("order_id")
            if n < 1:
                continue
            fees = fee(price, n)
        ledger.add(mode=mode, ticker=o["ticker"], game_id=o["game_id"], player_id=o["player_id"], player=o["name"],
                   stat=o["stat"], strike=o["strike"], side=o["side"], contracts=n, price=price, fees=fees,
                   p_model=o["p_model"], p_market=o["p_market"], p_final=o["p_final"], ev=o["ev"], order_id=oid)
        log(f"  [{mode}] BUY {n:>3} {o['side'].upper():3s} {o['name']} {o['stat']} {o['strike'] + 0.5:g}+ @ {price:.2f}"
            f"  model {o['p_model']:.2f} mkt {o['p_market']:.2f} -> EV {o['ev']:+.3f}/contract")
    if orders.empty:
        log("  no orders this pass")
    return orders


def settle(live: bool = False, log=print) -> int:
    mode = "live" if live else "paper"
    ledger, client, n = Ledger(), Kalshi("prod"), 0
    for t in ledger.trades(mode, "open").to_dict("records"):
        m = client.market(t["ticker"])
        if m.get("result") in ("yes", "no") or m.get("status") in ("settled", "finalized"):
            v = m.get("settlement_value_dollars")
            yes_value = float(v) if v not in (None, "") else (1.0 if m.get("result") == "yes" else 0.0)
            ledger.settle(t["id"], yes_value)
            n += 1
    log(f"  settled {n} {mode} trades")
    return n


# ----------------------------------------------------------------------------- model vs. the actual market
def settled_quotes(cfg: dict, frame: pd.DataFrame, min_volume: float = 25, log=print, refresh: bool = False) -> pd.DataFrame:
    """Every settled prop market Kalshi has published (recent + archive endpoints) with its last
    pre-kickoff bid/ask. `frame` must hold the played games of all seasons to cover. Cached on disk;
    only tickers that are not in the cache yet are downloaded."""
    cache = DATA / "kalshi_settled.parquet"
    old = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    if len(old) and not refresh:
        return old
    client = Kalshi("prod")
    raw = []
    for s in cfg["kalshi"]["series"]:
        raw += client.markets(s, "settled") + client.historical_markets(s)
    mk = parse(raw).drop_duplicates("ticker")
    log(f"  {len(mk)} settled prop markets listed")
    if len(old):
        mk = mk[~mk["ticker"].isin(old["ticker"])]
    mk = attach_players(mk, frame)
    kick = frame.drop_duplicates("game_id").set_index("game_id")["kickoff"].dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT") \
        .dt.tz_convert("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    secs = lambda t: (t - epoch) // pd.Timedelta("1s")
    mk = mk[mk["player_id"].notna() & mk["game_id"].notna()].copy()
    mk["kick_ts"] = secs(mk["game_id"].map(kick))
    mk = mk[mk["result"].isin(["yes", "no"]) & (mk["volume"] >= min_volume) & mk["kick_ts"].notna() &
            (secs(pd.to_datetime(mk["open_time"], utc=True, format="ISO8601")) < mk["kick_ts"])].copy()
    log(f"  {len(mk)} new settled markets with volume >= {min_volume} that were open before kickoff; pulling quotes")
    def last_quote(arr, ts):
        px = lambda c, side: (c.get(side) or {}).get("close_dollars") or (c.get(side) or {}).get("close")
        arr = [c for c in arr if c["end_period_ts"] <= ts and px(c, "yes_bid") and px(c, "yes_ask")]
        if not arr:
            return None
        c = arr[-1]
        return (float(px(c, "yes_bid")), float(px(c, "yes_ask")), float(c.get("volume_fp") or c.get("volume") or 0),
                int(c["end_period_ts"]))

    q = {}
    for ts, g in mk.groupby("kick_ts"):
        for tk, arr in client.candles(list(g["ticker"]), int(ts) - 6 * 3600, int(ts), 60).items():
            if (v := last_quote(arr, ts)) is not None:
                q[tk] = v
    # archived markets are not served by the batch endpoint: fetch them one by one, a few at a time
    rest = mk[~mk["ticker"].isin(q)]
    if len(rest):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        log(f"  {len(rest)} of them are archived; pulling those one ticker at a time")
        local = threading.local()

        def one(a):
            tk, ts = a
            if not hasattr(local, "c"):
                local.c = Kalshi("prod")
            try:
                return tk, last_quote(local.c.historical_candles(tk, int(ts) - 6 * 3600, int(ts), 60), ts)
            except KalshiError:
                return tk, None

        with ThreadPoolExecutor(3) as ex:
            for tk, v in ex.map(one, zip(rest["ticker"], rest["kick_ts"])):
                if v is not None:
                    q[tk] = v
    nan4 = (np.nan, np.nan, np.nan, np.nan)
    for i, col in enumerate(["yes_bid", "yes_ask", "last_candle_volume", "quote_ts"]):
        mk[col] = mk["ticker"].map(lambda t: q.get(t, nan4)[i])
    mk = mk.dropna(subset=["yes_bid", "yes_ask"])
    mk["no_ask"] = 1 - mk["yes_bid"]; mk["no_bid"] = 1 - mk["yes_ask"]
    mk = pd.concat([old, mk], ignore_index=True) if len(old) else mk
    mk.to_parquet(cache)
    return mk


def market_backtest(min_volume: float = 25, log=print) -> dict:
    """Score the walk-forward model against real pre-kickoff Kalshi quotes on settled markets from
    the current season, and replay the trading rules on them. The model used for these predictions
    was trained only on earlier seasons."""
    cfg = load_config()
    cur = current_season()
    oos = pd.read_parquet(REPORTS / "oos_predictions.parquet")
    oos = oos[oos["season"] == cur].rename(columns={"k": "strike", "p_cal": "p_model"})
    frame = build(load_all(), include_future=False)
    frame = frame[frame["season"] == cur]
    mk = settled_quotes(cfg, frame, min_volume, log)
    d = mk.merge(oos[["game_id", "player_id", "stat", "strike", "p_model", "mu", "week"]],
                 on=["game_id", "player_id", "stat", "strike"], how="inner")
    d = d.merge(frame[["game_id", "player_id", "name", "games_career", "inj_questionable", "kickoff"]],
                on=["game_id", "player_id"], how="left")
    d["y"] = (d["result"] == "yes").astype(int)
    d["yes_ask_size"] = d["no_ask_size"] = 1e9  # depth is unknown historically
    d = add_edges(d, cfg, now=pd.Timestamp("2000-01-01"))
    d = d.dropna(subset=["p_market"])
    ll = lambda p: float(-np.mean(d["y"] * np.log(np.clip(p, 1e-4, 1)) + (1 - d["y"]) * np.log(np.clip(1 - p, 1e-4, 1))))
    res = dict(markets=int(len(d)), games=int(d["game_id"].nunique()), weeks=sorted(int(w) for w in d["week"].unique()),
               logloss_model=ll(d["p_model"]), logloss_market=ll(d["p_market"]), logloss_blend=ll(d["p_final"]),
               brier_model=float(((d["p_model"] - d["y"]) ** 2).mean()), brier_market=float(((d["p_market"] - d["y"]) ** 2).mean()))
    # replay the rules: gates 1-4, best rung per player/stat/game, flat $10 stake
    bets = d[d["fails"] == ""].sort_values("ev", ascending=False).drop_duplicates(["game_id", "player_id", "stat"]).copy()
    if len(bets):
        bets["n"] = np.floor(10 / bets["cost"])
        win = np.where(bets["side"] == "yes", bets["y"], 1 - bets["y"])
        bets["pnl"] = bets["n"] * (win - bets["price"] - fee_per_contract(bets["price"]))
        bets["staked"] = bets["n"] * bets["cost"]
        bets["win"] = win
        by_game = bets.groupby("game_id")[["pnl", "staked"]].sum()
        rng = np.random.default_rng(0)
        boots = [(lambda s: s["pnl"].sum() / s["staked"].sum())(by_game.sample(len(by_game), replace=True, random_state=rng))
                 for _ in range(2000)]
        res["strategy"] = dict(bets=int(len(bets)), win_rate=float(win.mean()), avg_price=float(bets["price"].mean()),
                               staked=float(bets["staked"].sum()), pnl=float(bets["pnl"].sum()),
                               roi=float(bets["pnl"].sum() / bets["staked"].sum()),
                               roi_ci90=[float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
                               by_stat={k: dict(bets=int(len(g)), roi=float(g["pnl"].sum() / g["staked"].sum()))
                                        for k, g in bets.groupby("stat")})
    d.to_parquet(REPORTS / "market_backtest.parquet")
    (REPORTS / "market_backtest.json").write_text(json.dumps(res, indent=1))
    return res
