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


def cmd_train(a):
    from . import model as _m
    _m.OPTS["drop_feats"] = load_config()["model"].get("drop_feats") or _m.OPTS["drop_feats"]
    from .data import load_all
    from .features import build
    from .model import train
    df = build(load_all(refresh=a.refresh))
    from .config import DATA
    df.to_parquet(DATA / "frame.parquet")
    print(f"training on {len(df):,} player-games x {df.shape[1]} columns")
    train(df, backtest=not a.no_backtest)
    print("models saved to models/ ; backtest in reports/backtest.json")
    print("next: `market-backtest --refresh` re-scores against Kalshi prices and refits the model/market blend")
    if not a.no_backtest:
        cmd_fit_copula(a)


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


def cmd_weather_backfill(a):
    from . import weather
    from .data import load
    h = weather.history(load("schedules"))
    print(f"{len(h)} games with archived weather in data/raw/weather_hist.parquet")


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
        rk = r["risk"]
        print(f"       P(all hit) {rk['p_all_hit']:.3f} [calibration band {rk['p_lo']:.3f}-{rk['p_hi']:.3f}, MC ±{1.96 * rk['mc_se']:.3f}]"
              f"  if legs were independent {rk['p_indep_model']:.3f}  pessimistic (90% market) {rk['p_pessimistic']:.3f}")
        print(f"       P(>=1 leg fails) {rk['p_ge1_fail']:.3f}  P(>=2 fail) {rk['p_ge2_fail']:.3f}  P(all fail) {rk['p_all_fail']:.3f}"
              f" | games {rk['n_games']}, max legs/game {rk['max_legs_one_game']}, max same team {rk['max_same_team']},"
              f" opposing pairs {rk['n_opposing_pairs']}, max |rho| {rk['max_abs_rho']:.2f}")
        for pp in sorted(rk["pairs"], key=lambda d: -abs(d["rho"]))[:3]:
            if abs(pp["rho"]) >= 0.05:
                print(f"       corr {pp['rho']:+.2f}: P({pp['b'][:32]} | {pp['a'][:32]}) = {pp['p_b_given_a']:.2f} vs {pp['p_b']:.2f} alone")
        if i <= 3:
            alts = combos.rung_alternatives(res["pricer"], L, r["legs"], cfg)
            for alt in sorted(alts, key=lambda d: -d["growth"])[:4]:
                print(f"       alt: {alt['swap_in']:50s} instead of {alt['swap_out'][:30]:30s} -> P {alt['p_model']:.3f} ROI {alt['roi']:+.1%}")
    pf = res.get("portfolio") or {}
    if pf:
        print(f"\nIf you bet this shortlist every week for a season: expected P&L ${pf['exp_season_pnl']:+.0f}, "
              f"P(losing season) {pf['p_losing_season']:.0%}, median max drawdown {pf['median_max_drawdown']:.0%}, "
              f"95th pct {pf['p95_max_drawdown']:.0%} (assumes the model's probabilities are right)")
    cb = REPORTS / "combo_backtest.json"
    if cb.exists():
        h = json.loads(cb.read_text())
        print("history, random same-game combos (unseen season): " + ", ".join(
            f"{k} legs predicted {v['p_copula']:.3f} hit {v['hit']:.3f}" for k, v in h["by_legs"].items()))
    res["combos"].drop(columns=["risk"]).assign(desc=res["combos"]["desc"]).to_csv(REPORTS / "combos.csv", index=False)
    print("\nsaved reports/combos.csv. Build these in the Kalshi app's combo builder and compare the quote to 'accept up to'.")



def cmd_fit_realization(a):
    from . import strategy
    er = strategy.fit_realization()
    print(f"edge realization lambda (1 = the blend's disagreement with Kalshi is fully real, 0 = market right): "
          f"{er['pooled']:.2f} [10-90%: {er['pooled_lo']:.2f}..{er['pooled_hi']:.2f}] on {er['n']:,} quotes")
    print("by season: " + ", ".join(f"{k} {v:.2f}" for k, v in er["by_season"].items()))
    print(pd.DataFrame(er["by_stat"]).T.round(2).to_string())


def cmd_combo_bt_market(a):
    from . import strategy
    s = strategy.backtest(all_stats=a.all_stats)
    _show_combo_bt(s)


def _show_combo_bt(s: dict):
    rows = []
    for k, r in s.items():
        for h, v in r["by_hold"].items():
            rows.append(dict(legs=int(k), hold=float(h), combos=r["combos"], weeks=r["weeks"], p_model=r["p_model"],
                             p_fair=r["p_fair"], hit_rate=r["hit_rate"], avg_price=v["avg_price"], pred_roi=v["pred_roi"],
                             roi=v["roi"], roi_ci90=f"{v['roi_lo']:+.0%}..{v['roi_hi']:+.0%}", p_roi_pos=v["p_roi_pos"],
                             sd_per_bet=v["sd_per_bet"], max_dd_units=v["max_dd_units"]))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def cmd_combo_plan(a):
    from . import strategy
    t = strategy.plan(a.bankroll, a.weeks, a.per_week, a.target, a.hold, frac=a.kelly, all_stats=a.all_stats)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(f"bankroll ${a.bankroll:,.0f}, {a.weeks} weeks x {a.per_week} combos/week, target {a.target:+.0%}, "
          f"assumed combo hold {a.hold:.0%}, sizing {a.kelly:g} x Kelly\n")
    print(t.T.to_string())
    solid = t[t["combos_tested"] >= 40]
    solid = solid if len(solid) else t
    best = solid.loc[solid["kelly_season_ret"].idxmax()] if len(solid) else None
    if best is not None:
        print(f"\nmoney in -> money out for one {int(best['legs'])}-leg combo at {best['price']:.3f} "
              f"(hit rate {best['hit_rate']:.1%}):")
        print(strategy.stake_table(best["hit_rate"], best["price"]).round(2).to_string(index=False))


def cmd_combo_math(a):
    from . import strategy
    t = strategy.leg_count_math(a.p_leg, a.edge, a.hold, a.rho, bankroll=a.bankroll, frac=a.kelly, n_bets=a.n_bets)
    print(f"each leg: our P {a.p_leg:.0%}, market P {a.p_leg / (1 + a.edge):.1%} (edge {a.edge:+.0%}), leg correlation {a.rho}, "
          f"combo hold {a.hold:.0%}, bankroll ${a.bankroll:,.0f} at {a.kelly:g} Kelly, {a.n_bets} bets\n")
    print(t.round(4).to_string(index=False))


def cmd_combo_run(a):
    from . import combos, strategy
    from .bot import scan
    cfg = load_config()
    last = REPORTS / "last_scan.parquet"
    px = pd.read_parquet(last) if (not a.rescan and last.exists() and time.time() - last.stat().st_mtime < 1800) else scan(cfg)
    res = combos.suggest(px, cfg, top=20)
    if res["combos"].empty:
        print("no combos pass"); return
    open_ = strategy.open_combos()
    picks = strategy.choose(res, cfg, cfg["risk"]["bankroll"], float(open_["stake"].sum()) if len(open_) else 0.0)
    if picks.empty:
        print("nothing clears the robust EV / Kelly gates"); return
    strategy.record(picks)
    for p in picks.to_dict("records"):
        print(f"  [paper] {p['n_legs']} legs  {p['contracts']} contracts @ {p['price']:.3f} (accept up to {p['max_price']:.3f})"
              f"  P model {p['p_model']:.3f} robust {p['p_robust']:.3f}  EV {p['ev']:+.1%}  stake ${p['stake']:.2f}\n      {p['desc']}")


def cmd_combo_settle(a):
    from . import strategy
    strategy.settle()
    st = strategy.status()
    s = st[st["status"] == "settled"]
    print(f"{len(st)} paper combos, {len(s)} settled, P&L ${s['pnl'].sum():+.2f} on ${s['stake'].sum():.2f} staked"
          if len(st) else "no paper combos yet")


def _scan_cached(a):
    from .bot import scan
    last = REPORTS / "last_scan.parquet"
    if not getattr(a, "rescan", False) and last.exists() and time.time() - last.stat().st_mtime < 3600:
        return pd.read_parquet(last)
    px = scan(load_config()); px.to_parquet(last)
    return px


def cmd_ladder(a):
    from .ladder import STAT_ALIASES, ladder_view
    px = _scan_cached(a)
    stat = STAT_ALIASES.get(a.stat.lower(), a.stat)
    t = ladder_view(px, a.player, stat)
    if t.empty:
        print("no such ladder open on Kalshi right now"); return
    row = px[px["name"].str.contains(a.player, case=False, na=False) & (px["stat"] == stat)].iloc[0]
    print(f"{row['name']} {stat}: model projection {row['mu']:.1f}\n"
          "Each rung is its own $1 contract. market_% = mid price; lands_here = market chance of finishing between this\n"
          "rung and the next; payout_x = dollars back per $1 if YES hits (fee included); EV = expected profit per $1\n"
          "using the model's probability. The best rung is the highest EV, not the highest probability.\n")
    print(t.round(3).to_string(index=False))


def cmd_combo_price(a):
    from .ladder import price_combo
    px = _scan_cached(a)
    r = price_combo(px, a.legs, load_config())
    L = r["legs"]
    print("legs (market % = Kalshi mid; model % = what the bot believes):")
    for l in L.to_dict("records"):
        print(f"  {l['label']:42s} market {l['p_mkt']:.1%}  model {l['p_leg']:.1%}  ask {l['price_leg']:.2f}  [{l['ticker']} {l['side'].upper()}]")
    print(f"\nP(all legs hit):\n  if the legs were independent, at market prices  {r['p_ind_mkt']:.2%}  (what multiplying the prices gives)"
          f"\n  at market prices WITH correlation               {r['p_fair']:.2%}  -> fair price {r['p_fair']:.3f}, payout {1 / r['p_fair']:.2f}x"
          f"\n  the model's legs WITH correlation               {r['p_model']:.2%}  (calibrated for {len(L)} legs: {r['p_model_cal']:.2%})"
          f"\n  multiplying the asks                            {r['product_of_asks']:.2%}")
    rk = r["risk"]
    for p in sorted(rk["pairs"], key=lambda d: -abs(d["rho"])):
        if abs(p["rho"]) >= 0.03:
            print(f"  correlation {p['rho']:+.2f}: P({p['b']} | {p['a']}) = {p['p_b_given_a']:.0%} vs {p['p_b']:.0%} alone")
    print(f"\nexpected Kalshi quote ~{r['est_price']:.3f} (fair x (1 + the measured markup for {len(L)} legs)); "
          f"accept at most {r['max_price']:.3f}")
    b = r["bet"]
    verdict = "worth it" if b["ev_per_dollar"] > 0 else "NOT worth it"
    print(f"at that quote: payout {b['payout_x']:.2f}x, EV {b['ev_per_dollar']:+.1%} per $1 -> {verdict}; "
          f"Kelly {b['kelly']:.1%} of bankroll; P(>=1 leg misses) {rk['p_ge1_fail']:.0%}, P(>=2 miss) {rk['p_ge2_fail']:.0%}")


def cmd_tune_risk(a):
    from . import tuning
    r = tuning.run()
    print(f"walk-forward rules: {r['walk_forward']}\nshipped rules: {r['shipped_rules']}")
    print(r["table"][["frac", "cap_trade", "cap_week", "bets", "actual_return", "median_return", "p5_return", "p_loss",
                      "p95_max_dd", "usable"]].round(3).to_string(index=False))


def cmd_account(a):
    """Read-only look at the Kalshi account. Never places, amends or cancels anything."""
    from .kalshi import Kalshi, orders_unlocked
    k = Kalshi("prod")
    print(f"order lock: {'UNLOCKED (KALSHI_ALLOW_ORDERS=1)' if orders_unlocked() else 'LOCKED - only GET requests can be sent'}")
    if not k.has_key:
        print("no API key in this environment: set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY (or KALSHI_PRIVATE_KEY_PATH)")
        return
    snap = k.account_snapshot()
    (REPORTS / "account_snapshot.json").write_text(json.dumps(snap, indent=1, default=str))
    b = snap.get("balance", {})
    if "error" in b:
        print("balance:", b["error"]); return
    bal = float(b["balance_dollars"]) if "balance_dollars" in b else b.get("balance", 0) / 100
    print(f"cash balance ${bal:,.2f}  portfolio value {b.get('portfolio_value_dollars', b.get('portfolio_value', 'n/a'))}")
    pos = snap.get("positions", {})
    mp = pos.get("market_positions", []) if isinstance(pos, dict) else []
    print(f"open market positions: {len(mp)}; event positions: {len(pos.get('event_positions', []) if isinstance(pos, dict) else [])}")
    for x in mp[:25]:
        print(f"  {x.get('ticker'):55s} position {x.get('position_fp', x.get('position'))}  exposure {x.get('market_exposure_dollars', x.get('market_exposure'))}"
              f"  realized {x.get('realized_pnl_dollars', x.get('realized_pnl'))}")
    for name in ("orders", "fills", "settlements"):
        v = snap.get(name)
        print(f"{name}: {len(v) if isinstance(v, list) else v}")
    f = snap.get("fills") or []
    if isinstance(f, list) and f:
        df = pd.DataFrame(f)
        cols = [c for c in ("created_time", "ticker", "side", "action", "count_fp", "yes_price_dollars", "no_price_dollars", "is_taker") if c in df.columns]
        print(df[cols].head(15).to_string(index=False))
    st = snap.get("settlements") or []
    if isinstance(st, list) and st:
        sd = pd.DataFrame(st)
        print(f"settled markets: {len(sd)}; columns: {list(sd.columns)[:12]}")
    for name in ("rfqs", "quotes", "api_limits"):
        v = snap.get(name)
        print(f"{name}: {json.dumps(v, default=str)[:300]}")
    print("full snapshot saved to reports/account_snapshot.json (git-ignored)")


def cmd_params(a):
    from .params import coverage
    coverage(write=True)


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
        if a.combos:  # paper combos ride on the scan run_once just saved
            from . import strategy
            strategy.settle()
            a.rescan = False
            cmd_combo_run(a)
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
    print(json.dumps(market_backtest(min_volume=a.min_volume, refresh=a.refresh), indent=1))


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
    s.add_argument("--refresh", action="store_true")
    s.add_argument("--combos", action="store_true", help="also choose/size/record PAPER combos each pass"); s.set_defaults(f=cmd_run)
    s = sub.add_parser("settle", help="settle finished positions in the ledger"); s.add_argument("--live", action="store_true"); s.set_defaults(f=cmd_settle)
    s = sub.add_parser("status", help="bankroll, P&L and recent trades"); s.add_argument("--live", action="store_true")
    s.add_argument("-n", type=int, default=30); s.set_defaults(f=cmd_status)
    s = sub.add_parser("market-backtest", help="model vs real Kalshi pre-kickoff quotes; also refits the model/market blend")
    s.add_argument("--min-volume", type=float, default=25)
    s.add_argument("--refresh", action="store_true", help="also download markets settled since the last run")
    s.set_defaults(f=cmd_market_backtest)
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
    s.add_argument("--test-from", type=int, default=2026); s.add_argument("-n", type=int, default=4000); s.set_defaults(f=cmd_combo_backtest)
    sub.add_parser("weather-backfill", help="download historical game weather (slow, resumable)").set_defaults(f=cmd_weather_backfill)
    sub.add_parser("fit-realization", help="how much of the model's disagreement with Kalshi comes true").set_defaults(f=cmd_fit_realization)
    s = sub.add_parser("combo-backtest-market", help="combos by leg count on real 2025-26 Kalshi prices and outcomes")
    s.add_argument("--all-stats", action="store_true", help="include stats Kalshi won't let into combos"); s.set_defaults(f=cmd_combo_bt_market)
    s = sub.add_parser("combo-plan", help="stake sizing / expected returns / risk by leg count")
    s.add_argument("--bankroll", type=float, default=500); s.add_argument("--weeks", type=int, default=8)
    s.add_argument("--per-week", type=int, default=3); s.add_argument("--target", type=float, default=0.20)
    s.add_argument("--hold", type=float, default=0.10); s.add_argument("--kelly", type=float, default=0.25)
    s.add_argument("--all-stats", action="store_true", help="use the backtest that includes rec / rec yds / rush yds legs")
    s.set_defaults(f=cmd_combo_plan)
    s = sub.add_parser("combo-math", help="3 vs 5 legs etc. with same-quality legs: EV, variance, Kelly, growth")
    s.add_argument("--p-leg", type=float, default=0.60); s.add_argument("--edge", type=float, default=0.05)
    s.add_argument("--hold", type=float, default=0.10); s.add_argument("--rho", type=float, default=0.0)
    s.add_argument("--bankroll", type=float, default=500); s.add_argument("--kelly", type=float, default=0.25)
    s.add_argument("--n-bets", type=int, default=24); s.set_defaults(f=cmd_combo_math)
    s = sub.add_parser("combo-run", help="policy: choose, size and PAPER-record combos"); s.add_argument("--rescan", action="store_true")
    s.set_defaults(f=cmd_combo_run)
    sub.add_parser("combo-settle", help="settle paper combos leg by leg").set_defaults(f=cmd_combo_settle)
    s = sub.add_parser("ladder", help="one player's rung ladder: prices, implied %, payout, model %, EV per rung")
    s.add_argument("player"); s.add_argument("stat", help="rec yds, rec, rush yds, pass yds, pass td, td, ...")
    s.add_argument("--rescan", action="store_true"); s.set_defaults(f=cmd_ladder)
    s = sub.add_parser("combo-price", help='price your own combo: combo-price "Smith-Njigba rec yds 80+" "Mahomes pass td 2+"')
    s.add_argument("legs", nargs="+"); s.add_argument("--rescan", action="store_true"); s.set_defaults(f=cmd_combo_price)
    sub.add_parser("tune-risk", help="walk-forward rules + bankroll risk table on real quotes").set_defaults(f=cmd_tune_risk)
    sub.add_parser("account", help="READ-ONLY view of your Kalshi account (balance, positions, fills, settlements)").set_defaults(f=cmd_account)
    sub.add_parser("params", help="coverage of the 250-parameter checklist -> PARAMETERS.md").set_defaults(f=cmd_params)
    s = sub.add_parser("kill", help="engage (or --off release) the kill switch")
    s.add_argument("--off", action="store_true"); s.add_argument("--cancel", action="store_true", help="also cancel resting live orders")
    s.set_defaults(f=cmd_kill)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
