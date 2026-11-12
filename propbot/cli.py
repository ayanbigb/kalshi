"""Command line entry point:  python -m propbot <command>"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from .config import REPORTS, STATS, load_config

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)


def cmd_update(a):
    from .data import load_all
    raw = load_all(refresh=True)
    ps = raw["player_stats"]
    cur = ps["season"].max()
    print(f"data refreshed: {len(ps):,} player-weeks, latest = {cur} week {ps.loc[ps['season'] == cur, 'week'].max()}")


def _frame(refresh: bool = False, max_age_h: float = 6.0):
    """Feature frame, cached in data/frame.parquet for a few hours (building it takes minutes)."""
    from .config import DATA
    from .data import load_all
    from .features import build
    path = DATA / "frame.parquet"
    raw = load_all(refresh=refresh)
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < max_age_h * 3600:
        return pd.read_parquet(path), raw
    df = build(raw)
    df.to_parquet(path)
    return df, raw


def cmd_train(a):
    from .model import train
    df, _ = _frame(refresh=a.refresh, max_age_h=0)
    print(f"training on {len(df):,} player-games x {df.shape[1]} columns")
    train(df, backtest=not a.no_backtest)
    print("models saved to models/ ; backtest in reports/backtest.json")
    if not a.no_backtest:
        cmd_fit_copula(a)


def cmd_fit_copula(a):
    from . import combos
    oos = pd.read_parquet(REPORTS / "oos_predictions.parquet")
    c = combos.fit_copula(oos)
    combos.save_copula(c)
    big = sorted(((k, v) for k, v in c["rho"].items() if "*" not in k and c["n"].get(k, 0) >= 500),
                 key=lambda t: -abs(t[1]))[:15]
    print(f"copula fitted on {len(oos):,} out-of-sample rows -> models/copula.json; strongest pairs:")
    for k, v in big:
        print(f"  {k:45s} {v:+.2f}  (n={c['n'][k]})")


def cmd_combo_backtest(a):
    from . import combos
    oos = pd.read_parquet(REPORTS / "oos_predictions.parquet")
    print(json.dumps(combos.backtest_copula(oos, a.test_from, a.n), indent=1))


def cmd_matchup(a):
    from .model import PropModel
    from .report import matchup
    df, raw = _frame(refresh=a.refresh)
    matchup(df, raw, a.player, PropModel())


def cmd_combos(a):
    from . import combos
    from .bot import scan
    cfg = load_config()
    c = cfg["combos"]
    if a.max_legs: c["max_legs"] = a.max_legs
    if a.min_legs: c["min_legs"] = a.min_legs
    if a.same_game: c["same_game_only"] = True
    if a.yes_only: c["allow_no"] = False
    last = REPORTS / "last_scan.parquet"
    if not a.rescan and last.exists() and time.time() - last.stat().st_mtime < a.max_age * 60:
        px = pd.read_parquet(last); print(f"using {last.name} ({(time.time() - last.stat().st_mtime) / 60:.0f} min old)")
    else:
        px = scan(cfg); px.to_parquet(last)
    if a.team:
        t = a.team.upper(); px = px[(px["home"] == t) | (px["away"] == t)]
    if a.player:
        keep = px.loc[px["name"].str.contains(a.player, case=False, na=False), "game_id"].unique()
        px = px[px["game_id"].isin(keep)]
    res = combos.suggest(px, cfg, top=a.top)
    if res["combos"].empty:
        print("no combo clears the edge / ROI thresholds right now"); return
    print("\nBest combo at each size (how many legs):")
    b = res["best_by_size"]
    print(b[["n_legs", "p_model", "p_fair", "p_indep", "est_price", "max_price", "roi", "growth", "stake"]].round(4).to_string(index=False))
    print("\nShortlist (diverse, ranked by expected log growth):")
    L = res["legs"]
    for i, r in enumerate(res["combos"].to_dict("records"), 1):
        print(f"\n#{i}  {r['n_legs']} legs | P(model) {r['p_model']:.3f}  fair {r['p_fair']:.3f}  independent {r['p_indep']:.3f}"
              f" | expect quote ~{r['est_price']:.3f}, accept up to {r['max_price']:.3f} | ROI {r['roi']:+.1%}  stake ${r['stake']:.2f}")
        for j in r["legs"]:
            l = L.loc[j]
            print(f"     - {l['label']:55s} p {l['p_leg']:.2f} vs mkt {l['p_mkt']:.2f}   [{l['ticker']} {l['side'].upper()}]")
        if i <= 3:
            alts = combos.rung_alternatives(res["pricer"], L, r["legs"], cfg)
            for alt in sorted(alts, key=lambda d: -d["growth"])[:4]:
                print(f"       alt: {alt['swap_in']:50s} instead of {alt['swap_out'][:30]:30s} -> P {alt['p_model']:.3f} ROI {alt['roi']:+.1%}")
    res["combos"].assign(desc=res["combos"]["desc"]).to_csv(REPORTS / "combos.csv", index=False)
    print("\nsaved reports/combos.csv. Build these in the Kalshi app's combo builder and compare the quote to 'accept up to'.")


def cmd_report(a):
    rep = json.loads((REPORTS / "backtest.json").read_text())
    rows = [dict(stat=k, n=v["model"]["n"], logloss=v["model"]["logloss"], naive_logloss=v["baseline"]["logloss"],
                 brier=v["model"]["brier"], auc_tradeable=v["model_tradeable"]["auc"],
                 acc_tradeable=v["model_tradeable"]["acc"], naive_acc=v["baseline_tradeable"]["acc"],
                 calib_err=v["ece_cal"], proj_mae=v["mae_projection"]) for k, v in rep.items()]
    print("Walk-forward backtest (each season predicted by models trained only on earlier seasons)\n")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    p = REPORTS / "market_backtest.json"
    if p.exists():
        print("\nModel vs. real Kalshi pre-kickoff quotes:\n" + p.read_text())


def cmd_project(a):
    from .data import load_all
    from .features import build
    from .model import PropModel
    df = build(load_all())
    fut = df[df["is_future"]]
    fut = fut[fut["kickoff"] == fut.groupby("team")["kickoff"].transform("min")]
    m = PropModel()
    out = fut[["name", "position", "team", "opp", "kickoff"]].copy()
    for stat, s in STATS.items():
        ok = fut["position"].isin(s["pos"]) & (fut["is_primary_qb"] if s.get("starter_qb") else True)
        out.loc[ok, stat] = m.project(fut[ok], stat)
    if a.player:
        out = out[out["name"].str.contains(a.player, case=False, na=False)]
    if a.team:
        out = out[out["team"] == a.team.upper()]
    out = out.sort_values(["kickoff", "team", "rr_yds"], ascending=[True, True, False])
    print(out.round(1).to_string(index=False))


def _show(px: pd.DataFrame, top: int, all_rows: bool):
    if px.empty:
        print("nothing priced"); return
    d = px if all_rows else px[px["fails"] == ""]
    d = d.sort_values("ev", ascending=False).head(top)
    cols = ["name", "team", "stat", "strike", "mu", "p_model", "p_market", "p_final", "side", "price", "ev", "roi", "spread", "size", "fails"]
    d = d.assign(strike=d["strike"] + 0.5)[cols].rename(columns={"strike": "N+", "mu": "proj"})
    print(d.round(3).to_string(index=False) if len(d) else "no market passes the gates right now")


def cmd_scan(a):
    from .bot import scan
    px = scan(refresh=a.refresh)
    if len(px):
        px.to_parquet(REPORTS / "last_scan.parquet")
    _show(px, a.top, a.all)


def cmd_run(a):
    from .bot import run_once, settle
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LIVE" if a.live else "paper")
        settle(a.live)
        run_once(live=a.live, refresh=a.refresh)
        if not a.loop:
            break
        time.sleep(a.loop * 60)


def cmd_settle(a):
    from .bot import settle
    settle(a.live)


def cmd_status(a):
    from .ledger import Ledger
    mode = "live" if a.live else "paper"
    L = Ledger()
    print(json.dumps(L.summary(mode, load_config()["risk"]["bankroll"]), indent=1))
    t = L.trades(mode)
    if len(t):
        print(t[["ts", "player", "stat", "strike", "side", "contracts", "price", "p_model", "p_market", "ev", "status", "pnl"]]
              .tail(a.n).round(3).to_string(index=False))


def cmd_market_backtest(a):
    from .bot import market_backtest
    print(json.dumps(market_backtest(min_volume=a.min_volume), indent=1))


def cmd_kill(a):
    from .risk import KILL_FILE
    if a.off:
        KILL_FILE.unlink(missing_ok=True); print("kill switch released")
    else:
        KILL_FILE.write_text("stop"); print("kill switch engaged: no new orders will be placed")
        if a.cancel:
            from .kalshi import Kalshi
            Kalshi(load_config()["kalshi"]["env"]).cancel_all(); print("resting orders cancelled")


def main():
    p = argparse.ArgumentParser(prog="propbot", description="Kalshi NFL player-prop model and bot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update", help="refresh current-season nflverse data").set_defaults(f=cmd_update)
    s = sub.add_parser("train", help="build features, walk-forward backtest, fit final models")
    s.add_argument("--refresh", action="store_true"); s.add_argument("--no-backtest", action="store_true"); s.set_defaults(f=cmd_train)
    sub.add_parser("report", help="print backtest metrics").set_defaults(f=cmd_report)
    s = sub.add_parser("project", help="projections for the upcoming slate")
    s.add_argument("--player"); s.add_argument("--team"); s.set_defaults(f=cmd_project)
    s = sub.add_parser("scan", help="price open Kalshi prop markets (places nothing)")
    s.add_argument("--top", type=int, default=40); s.add_argument("--all", action="store_true", help="include markets that fail gates")
    s.add_argument("--refresh", action="store_true"); s.set_defaults(f=cmd_scan)
    s = sub.add_parser("run", help="scan + risk engine + orders (paper unless --live)")
    s.add_argument("--live", action="store_true"); s.add_argument("--loop", type=int, default=0, help="repeat every N minutes")
    s.add_argument("--refresh", action="store_true"); s.set_defaults(f=cmd_run)
    s = sub.add_parser("settle", help="settle finished positions in the ledger"); s.add_argument("--live", action="store_true"); s.set_defaults(f=cmd_settle)
    s = sub.add_parser("status", help="bankroll, P&L and recent trades"); s.add_argument("--live", action="store_true")
    s.add_argument("-n", type=int, default=30); s.set_defaults(f=cmd_status)
    s = sub.add_parser("market-backtest", help="model vs real Kalshi pre-kickoff quotes this season")
    s.add_argument("--min-volume", type=float, default=25); s.set_defaults(f=cmd_market_backtest)
    s = sub.add_parser("matchup", help="every factor for one player's next game (history vs opp, weather, defenders...)")
    s.add_argument("player"); s.add_argument("--refresh", action="store_true"); s.set_defaults(f=cmd_matchup)
    s = sub.add_parser("combos", help="suggest multi-leg combos: how many legs and which rung of each ladder")
    s.add_argument("--top", type=int, default=10); s.add_argument("--max-legs", type=int); s.add_argument("--min-legs", type=int)
    s.add_argument("--same-game", action="store_true"); s.add_argument("--yes-only", action="store_true")
    s.add_argument("--team"); s.add_argument("--player", help="only games involving this player")
    s.add_argument("--rescan", action="store_true"); s.add_argument("--max-age", type=float, default=30, help="minutes a saved scan stays usable")
    s.set_defaults(f=cmd_combos)
    sub.add_parser("fit-copula", help="re-fit leg correlations from out-of-sample residuals").set_defaults(f=cmd_fit_copula)
    s = sub.add_parser("combo-backtest", help="does the correlation model price combos better than independence?")
    s.add_argument("--test-from", type=int, default=2025); s.add_argument("-n", type=int, default=6000); s.set_defaults(f=cmd_combo_backtest)
    s = sub.add_parser("kill", help="engage (or --off release) the kill switch")
    s.add_argument("--off", action="store_true"); s.add_argument("--cancel", action="store_true", help="also cancel resting live orders")
    s.set_defaults(f=cmd_kill)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
