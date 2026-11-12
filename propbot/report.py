"""Human-readable matchup report: every factor the model sees for one player's next game."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import defense
from .config import STATS
from .features import team_games

SHOW = ["receptions", "targets", "receiving_yards", "carries", "rushing_yards", "attempts", "passing_yards",
        "passing_tds", "td"]


def _pct(frame: pd.DataFrame, col: str, v: float, higher_is_tougher: bool) -> str:
    s = frame[col].dropna()
    if not len(s) or v != v:
        return "   n/a"
    p = (s < v).mean() if higher_is_tougher else (s > v).mean()
    return f"{100 * p:5.0f}%"


def matchup(frame: pd.DataFrame, raw: dict, name: str, model=None, out=print):
    fut = frame[frame["is_future"] & frame["name"].str.contains(name, case=False, na=False)]
    if fut.empty:
        out(f"no upcoming game found for '{name}'"); return
    r = fut.sort_values("kickoff").iloc[0]
    pid, opp, team = r["player_id"], r["opp"], r["team"]
    out(f"\n=== {r['name']} ({r['position']}, {team}) vs {opp} — {r['kickoff']:%a %b %d %I:%M %p} ET ===")
    out(f"line: spread {r['spread']:+.1f}  total {r['total_line']}  implied team total {r['implied_total']:.1f}  "
        f"{'home' if r['is_home'] == 1 else 'away'}  rest {r['rest']:.0f}d (edge {r.get('rest_edge', np.nan):+.0f})")
    out(f"record: {team} win% {r['rec_wpct']:.3f} (shrunk), pt diff/g {r['rec_pd_pg']:+.1f}, streak {r['rec_streak']:+.0f} | "
        f"{opp} win% {r['opp_rec_wpct']:.3f}, pt diff/g {r['opp_rec_pd_pg']:+.1f}, streak {r['opp_rec_streak']:+.0f}")
    wx = "indoors" if r["indoors"] == 1 else (f"{r['wx_temp']:.0f}F, wind {r['wx_wind']:.0f} mph (gusts {r['wx_gust']:.0f}), "
                                               f"precip {r['wx_precip']:.2f}in, snow {r['wx_snow']:.2f}in")
    out(f"weather: {wx}   travel {r['travel_km']:.0f} km, tz shift {r['tz_shift']:+.1f}h, altitude {r['altitude']:.0f} m"
        f"{'  PRIMETIME' if r['primetime'] else ''}{'  short week' if r['short_week'] else ''}"
        f"{'  body-clock spot' if r['body_clock'] else ''}")
    out(f"injury: {'Questionable' if r['inj_questionable'] else 'not listed / cleared'}"
        f"   vacated team targets {r['vac_tgt']:.2f}, carries {r['vac_carry']:.2f}   own OL missing {r['ol_miss']:.2f} starters")

    # ---- projections
    if model is not None:
        out("\nprojection (model mean) and P(N+) on a few rungs:")
        for stat, s in STATS.items():
            if r["position"] not in s["pos"] or (s.get("starter_qb") and not r["is_primary_qb"]):
                continue
            row = fut.iloc[[0]]
            mu = float(model.project(row, stat)[0])
            ks = [k for k in s["grid"] if abs(k - mu) <= max(1.5, 0.6 * mu)][:6] or s["grid"][:4]
            pr = model.prob(row, stat, [np.array(ks)])
            rungs = "  ".join(f"{k + 0.5:g}+:{p:.2f}" for k, p in zip(pr["k"], pr["p"]))
            out(f"  {stat:9s} {mu:7.1f}   {rungs}")

    # ---- history vs this franchise
    hist = frame[(frame["player_id"] == pid) & ~frame["is_future"]]
    vs = hist[hist["opp"] == opp]
    out(f"\nvs {opp}: {len(vs)} prior games  (avg over/under his own rolling average: "
        f"rec yds {r['vsopp_res_rec_yds']:+.1f}, rush yds {r['vsopp_res_rush_yds']:+.1f}, "
        f"pass yds {r['vsopp_res_pass_yds']:+.1f}, TD {r['vsopp_res_td']:+.2f}; shrunk)")
    if len(vs):
        cols = ["season", "week", "team"] + [c for c in SHOW if hist[c].sum() > 0]
        out(vs[cols].tail(10).to_string(index=False))
    out(f"vs former team: {'YES' if r['vs_former_team'] else 'no'}   "
        f"vs this head coach: {r['vscoach_n']:.0f} games, rec yds {r['vscoach_res_rec_yds']:+.1f}, "
        f"rush yds {r['vscoach_res_rush_yds']:+.1f}")

    # ---- opposing defense
    tgh = frame[~frame["is_future"]]
    out(f"\n{opp} defense (expected lineup, snap-weighted; percentile = how tough vs league):")
    for col, lab, tough in (("opp_u_cb_ypt", "CB yards/target allowed", False), ("opp_u_s_ypt", "S yards/target allowed", False),
                            ("opp_u_lb_ypt", "LB yards/target allowed", False), ("opp_u_cb1_ypt", "top CB yards/target", False),
                            ("opp_u_db_tdr", "DB TD rate allowed", False), ("opp_u_press_pg", "front pressures/game", True),
                            ("opp_u_def_mtk", "missed-tackle rate", False),
                            ("opp_pa_rec_yds_e5", f"rec yds allowed to {r['position']}s (EWM)", False),
                            ("opp_pts_allowed_e5", "points allowed (EWM)", False)):
        if col in r.index:
            out(f"  {lab:34s} {r[col]:7.2f}   tougher than {_pct(tgh, col, r[col], tough)} of defenses")
    out(f"  missing regulars (snap share): CB {r['opp_u_miss_cb']:.2f}  S {r['opp_u_miss_s']:.2f}  "
        f"LB {r['opp_u_miss_lb']:.2f}  DL {r['opp_u_miss_dl']:.2f}")

    # ---- the defenders, and how he has done against each
    try:
        tg = team_games(raw["schedules"])
        sn = defense._defender_form(defense._snaps(raw, tg))
        lu = defense.opponent_lineup(raw, frame, tg, int(r["season"]), r["game_id"], opp)
    except Exception as e:
        out(f"  (defender detail unavailable: {e})"); return
    played = sn[(sn["share"] >= 0.3)][["pid", "game_id"]]
    mine = hist[["game_id", "res_rec_yds", "res_rush_yds", "res_pass_yds", "receiving_yards", "rushing_yards"]]
    pair = played.merge(mine, on="game_id")
    ph = pair.groupby("pid").agg(g=("game_id", "size"), res_rec=("res_rec_yds", "mean"), res_rush=("res_rush_yds", "mean"),
                                 rec=("receiving_yards", "mean"), rush=("rushing_yards", "mean"))
    lu = lu.merge(ph, left_on="pid", right_index=True, how="left")
    lu = lu[lu["w"] >= 0.2]
    out(f"\nexpected {opp} defenders (share = expected snap %) and {r['name']}'s games against each:")
    show = lu[["grp", "name", "w", "ypt", "cmp", "press_pg", "g", "res_rec", "res_rush"]].rename(columns={
        "w": "share", "ypt": "ypt_alw", "cmp": "cmp%_alw", "press_pg": "press/g", "g": "games_vs",
        "res_rec": "rec_yds_vs_avg", "res_rush": "rush_yds_vs_avg"})
    out(show.round(2).to_string(index=False))
    out(f"summary vs these DBs: {r.get('vsdb_n', np.nan):.1f} games each on avg, rec yds {r.get('vsdb_res_rec_yds', np.nan):+.1f}; "
        f"vs this front: rush yds {r.get('vsfront_res_rush_yds', np.nan):+.1f}")
