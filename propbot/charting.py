"""Charting and participation data (nflverse, free).

FTN charting (2022+, published during the season, so usable live) -> rolling, pre-game features:
  receivers  first-read target share, catchable / contested / created-reception rates, screen-target share
  QBs        play-action, out-of-pocket, interception-worthy, throwaway, first-read and checkdown rates
  rushers    average defenders in the box, RPO share
  defenses   blitz rate, average pass rushers, average box against the run, play-action allowed

Participation (2016-2025; published after each season, NOT during it) -> SEASON-LAGGED traits only, so a
2026 game sees exactly what a 2024 game saw (the previous seasons), never a stale in-season number:
  route participation, targets and yards per route run (career through last season)
  yards per target vs man and vs zone (career through last season) and the opponent defense's man rate last
  season, combined into a matchup term
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FTN_PLAYER = ["ftn_tgt", "ftn_first_read", "ftn_catchable", "ftn_contested", "ftn_created", "ftn_screen_tgt",
              "ftn_db", "ftn_pa", "ftn_oop", "ftn_intw", "ftn_throwaway", "ftn_fr_throw", "ftn_chk_throw",
              "ftn_rush", "ftn_box_sum", "ftn_rpo"]


def _ewm_prior(df: pd.DataFrame, key: str, cols: list[str], hl: float) -> pd.DataFrame:
    s = df.groupby(key, sort=False)[cols].shift(1)
    out = s.groupby(df[key], sort=False).transform(lambda x: x.ewm(halflife=hl, min_periods=1, ignore_na=True).mean())
    return out


def ftn_features(b: pd.DataFrame, raw: dict, tg: pd.DataFrame) -> pd.DataFrame:
    f, pbp = raw.get("ftn"), raw["pbp"]
    if f is None or f.empty or "play_id" not in pbp.columns:
        return b
    f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f["play_id"] = f["play_id"].astype(float)
    p = pbp[["game_id", "play_id", "posteam", "defteam", "pass", "rush", "qb_dropback", "receiver_player_id",
             "rusher_player_id", "passer_player_id"]].merge(f, on=["game_id", "play_id"], how="inner")
    tb = lambda c: p[c].astype("boolean").fillna(False).astype(float)
    p["fr"] = (p["read_thrown"] == "1").astype(float)
    p["chk"] = (p["read_thrown"] == "CHK").astype(float)
    tgt = p[p["receiver_player_id"].notna()]
    rec = tgt.assign(catch_=tb("is_catchable_ball")[tgt.index], cont=tb("is_contested_ball")[tgt.index],
                     cre=tb("is_created_reception")[tgt.index], scr=tb("is_screen_pass")[tgt.index]) \
        .groupby(["game_id", "receiver_player_id"]).agg(ftn_tgt=("play_id", "size"), ftn_first_read=("fr", "sum"),
                                                         ftn_catchable=("catch_", "sum"), ftn_contested=("cont", "sum"),
                                                         ftn_created=("cre", "sum"), ftn_screen_tgt=("scr", "sum")) \
        .reset_index().rename(columns={"receiver_player_id": "player_id"})
    db = p[(p["qb_dropback"] == 1) & p["passer_player_id"].notna()]
    qb = db.assign(pa=tb("is_play_action")[db.index], oop=tb("is_qb_out_of_pocket")[db.index],
                   iw=tb("is_interception_worthy")[db.index], ta=tb("is_throw_away")[db.index]) \
        .groupby(["game_id", "passer_player_id"]).agg(ftn_db=("play_id", "size"), ftn_pa=("pa", "sum"), ftn_oop=("oop", "sum"),
                                                       ftn_intw=("iw", "sum"), ftn_throwaway=("ta", "sum"),
                                                       ftn_fr_throw=("fr", "sum"), ftn_chk_throw=("chk", "sum")) \
        .reset_index().rename(columns={"passer_player_id": "player_id"})
    ru = p[(p["rush"] == 1) & p["rusher_player_id"].notna()]
    rush = ru.assign(rpo=tb("is_rpo")[ru.index]).groupby(["game_id", "rusher_player_id"]).agg(
        ftn_rush=("play_id", "size"), ftn_box_sum=("n_defense_box", "sum"), ftn_rpo=("rpo", "sum")) \
        .reset_index().rename(columns={"rusher_player_id": "player_id"})
    pg = rec.merge(qb, on=["game_id", "player_id"], how="outer").merge(rush, on=["game_id", "player_id"], how="outer")
    have_ftn = set(f["game_id"].unique())
    b = b.merge(pg, on=["game_id", "player_id"], how="left")
    played_ftn = b["game_id"].isin(have_ftn) & ~b["is_future"]
    b.loc[played_ftn, FTN_PLAYER] = b.loc[played_ftn, FTN_PLAYER].fillna(0.0)
    b = b.sort_values(["player_id", "kickoff"]).reset_index(drop=True)
    e = _ewm_prior(b, "player_id", FTN_PLAYER, 6)
    den = lambda c: e[c].replace(0, np.nan)
    b["ftn_first_read_share_e6"] = e["ftn_first_read"] / den("ftn_tgt")
    b["ftn_first_read_pg_e6"] = e["ftn_first_read"]
    b["ftn_catchable_rate_e6"] = e["ftn_catchable"] / den("ftn_tgt")
    b["ftn_contested_rate_e6"] = e["ftn_contested"] / den("ftn_tgt")
    b["ftn_created_rate_e6"] = e["ftn_created"] / den("ftn_tgt")
    b["ftn_screen_share_e6"] = e["ftn_screen_tgt"] / den("ftn_tgt")
    b["ftn_pa_rate_e6"] = e["ftn_pa"] / den("ftn_db")
    b["ftn_oop_rate_e6"] = e["ftn_oop"] / den("ftn_db")
    b["ftn_intw_rate_e6"] = e["ftn_intw"] / den("ftn_db")
    b["ftn_throwaway_rate_e6"] = e["ftn_throwaway"] / den("ftn_db")
    b["ftn_fr_throw_rate_e6"] = e["ftn_fr_throw"] / den("ftn_db")
    b["ftn_chk_rate_e6"] = e["ftn_chk_throw"] / den("ftn_db")
    b["ftn_box_avg_e6"] = e["ftn_box_sum"] / den("ftn_rush")
    b["ftn_rpo_share_e6"] = e["ftn_rpo"] / den("ftn_rush")
    # ---- defense tendencies
    d = p.assign(blitz=(p["n_blitzers"].fillna(0) > 0).astype(float), pa=tb("is_play_action"))
    dbd = d[d["qb_dropback"] == 1].groupby(["game_id", "defteam"]).agg(dblitz=("blitz", "mean"), drushers=("n_pass_rushers", "mean"),
                                                                       dpa=("pa", "mean"))
    rbd = d[d["rush"] == 1].groupby(["game_id", "defteam"]).agg(dbox=("n_defense_box", "mean"))
    D = dbd.join(rbd, how="outer").reset_index().rename(columns={"defteam": "opp"})
    D = tg[["game_id", "team", "kickoff"]].rename(columns={"team": "opp"}).merge(D, on=["game_id", "opp"], how="left") \
        .sort_values(["opp", "kickoff"]).reset_index(drop=True)
    cols = ["dblitz", "drushers", "dpa", "dbox"]
    De = _ewm_prior(D, "opp", cols, 5)
    De.columns = [f"opp_ftn_{c}_e5" for c in cols]
    D = pd.concat([D[["game_id", "opp"]], De], axis=1)
    b = b.merge(D, on=["game_id", "opp"], how="left")
    return b.drop(columns=FTN_PLAYER)


def participation_features(b: pd.DataFrame, raw: dict) -> pd.DataFrame:
    part, pbp = raw.get("participation"), raw["pbp"]
    if part is None or part.empty or "play_id" not in pbp.columns:
        return b
    part = part.rename(columns={"nflverse_game_id": "game_id"})
    part["play_id"] = part["play_id"].astype(float)
    p = pbp[["game_id", "play_id", "season", "posteam", "defteam", "qb_dropback", "receiver_player_id", "receiving_yards",
             "complete_pass", "sack"]].merge(part[["game_id", "play_id", "offense_players", "defense_man_zone_type"]],
                                             on=["game_id", "play_id"], how="inner")
    db = p[(p["qb_dropback"] == 1)].copy()
    # routes: every non-QB skill player on the field on a dropback (OL / QB filtered by the frame's positions later)
    on = db[["game_id", "season", "posteam", "offense_players"]].assign(pid=db["offense_players"].astype(str).str.split(";")) \
        .explode("pid")
    on = on[on["pid"].str.len() > 5]
    routes = on.groupby(["season", "pid"]).size().rename("routes")
    team_db = db.groupby(["season", "posteam"]).size().rename("team_db")
    main_team = on.groupby(["season", "pid"])["posteam"].agg(lambda s: s.mode().iloc[0]).rename("posteam")
    R = pd.concat([routes, main_team], axis=1).reset_index().merge(team_db.reset_index(), on=["season", "posteam"], how="left")
    tg_ = db[db["receiver_player_id"].notna() & (db["sack"] != 1)]
    T = tg_.groupby(["season", "receiver_player_id"]).agg(tgts=("play_id", "size"), yds=("receiving_yards", "sum")) \
        .reset_index().rename(columns={"receiver_player_id": "pid"})
    R = R.merge(T, on=["season", "pid"], how="left").fillna({"tgts": 0, "yds": 0})
    # career through the previous season (season-lagged)
    R = R.sort_values(["pid", "season"])
    for c in ("routes", "tgts", "yds", "team_db"):
        R[f"cum_{c}"] = R.groupby("pid")[c].cumsum()
    R["season"] = R["season"] + 1  # a season's traits are known from the next season on
    R["route_part_prev"] = R["routes"] / R["team_db"]
    R["tprr_car"] = (R["cum_tgts"] + 0.18 * 100) / (R["cum_routes"] + 100)
    R["yprr_car"] = (R["cum_yds"] + 1.4 * 100) / (R["cum_routes"] + 100)
    b = b.merge(R[["season", "pid", "route_part_prev", "tprr_car", "yprr_car"]].rename(columns={"pid": "player_id"}),
                on=["season", "player_id"], how="left")
    # man / zone: defenses last season, receivers career through last season
    mz = db[db["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"])].copy()
    mz["man"] = (mz["defense_man_zone_type"] == "MAN_COVERAGE").astype(float)
    dm = mz.groupby(["season", "defteam"])["man"].mean().rename("opp_man_rate_prev").reset_index()
    dm["season"] += 1
    b = b.merge(dm.rename(columns={"defteam": "opp"}), on=["season", "opp"], how="left")
    tz = mz[mz["receiver_player_id"].notna()]
    S = tz.groupby(["season", "receiver_player_id", "man"]).agg(t=("play_id", "size"), y=("receiving_yards", "sum")) \
        .unstack("man").fillna(0)
    S.columns = [f"{a}_{'man' if m else 'zone'}" for a, m in S.columns]
    S = S.reset_index().rename(columns={"receiver_player_id": "pid"}).sort_values(["pid", "season"])
    for c in ("t_man", "y_man", "t_zone", "y_zone"):
        S[c] = S.groupby("pid")[c].cumsum()
    S["season"] += 1
    k = 25.0
    S["ypt_man_car"] = (S["y_man"] + 7.5 * k) / (S["t_man"] + k)
    S["ypt_zone_car"] = (S["y_zone"] + 7.5 * k) / (S["t_zone"] + k)
    S["man_minus_zone"] = S["ypt_man_car"] - S["ypt_zone_car"]
    # carry each player's latest career split forward to later seasons he has no new data for
    b = b.merge(S[["season", "pid", "ypt_man_car", "ypt_zone_car", "man_minus_zone"]].rename(columns={"pid": "player_id"}),
                on=["season", "player_id"], how="left")
    b = b.sort_values(["player_id", "kickoff"])
    for c in ("route_part_prev", "tprr_car", "yprr_car", "ypt_man_car", "ypt_zone_car", "man_minus_zone"):
        b[c] = b.groupby("player_id")[c].ffill()
    b["man_zone_matchup"] = b["man_minus_zone"] * (b["opp_man_rate_prev"] - 0.45)  # ~league-average man rate
    return b


def add(b: pd.DataFrame, raw: dict, tg: pd.DataFrame, log=print) -> pd.DataFrame:
    for name, fn in (("ftn", lambda x: ftn_features(x, raw, tg)), ("participation", lambda x: participation_features(x, raw))):
        try:
            b = fn(b)
        except Exception as e:
            log(f"  [charting] {name} skipped: {type(e).__name__}: {e}")
    return b
