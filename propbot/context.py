"""Game-context and history features that sit on top of the rolling player form.

  team record      season-to-date wins / win% / point differential / recent form, for both teams
  vs opponent      the player's prior games against this franchise: count and average
                   over/under-performance relative to his own rolling average at the time
  vs former team   playing a team he used to play for
  vs coach         over/under-performance against the opposing head coach's defenses (follows coaches
                   across teams)
  referee          crew's history of game totals relative to the closing total
  schedule spot    primetime, Thursday, short week, rest edge, travel distance, time-zone shift,
                   body-clock (west-coast team at an early eastern kickoff), altitude
  weather          see weather.py (forecast for upcoming games, reanalysis for history)

Every number uses only games before kickoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import weather

RES_STATS = {"receiving_yards": "rec_yds", "receptions": "rec", "rushing_yards": "rush_yds",
             "passing_yards": "pass_yds", "td": "td", "rr_yds": "rr_yds", "carries": "car", "targets": "tgt"}


def add_residuals(b: pd.DataFrame) -> pd.DataFrame:
    """res_<stat> = actual - the player's pre-game rolling average (NaN for upcoming games)."""
    for col, short in RES_STATS.items():
        b[f"res_{short}"] = (b[col] - b[f"{col}_e6"]).where(~b["is_future"])
    return b


def _prior_split(b: pd.DataFrame, key: list[str], prefix: str, shrink: float = 2.0) -> pd.DataFrame:
    """Prior-games count and shrunk mean residual for every res_* column, grouped by `key`."""
    rcols = [f"res_{s}" for s in RES_STATS.values()]
    d = b[key + ["kickoff"] + rcols].sort_values("kickoff", kind="stable")
    k = d[key].astype(str).agg("|".join, axis=1)
    has = d[rcols[0]].notna().astype(float)
    n = has.groupby(k, sort=False).cumsum() - has
    out = pd.DataFrame(index=d.index)
    out[f"{prefix}_n"] = n
    for c in rcols:
        v = d[c].fillna(0)
        out[f"{prefix}_{c}"] = (v.groupby(k, sort=False).cumsum() - v) / (n + shrink)
    return out.reindex(b.index)


def team_record(tg: pd.DataFrame) -> pd.DataFrame:
    t = tg[["game_id", "team", "season", "kickoff", "pts", "pts_allowed"]].sort_values(["team", "kickoff"]).copy()
    played = t["pts"].notna()
    t["w"] = np.where(played, (t["pts"] > t["pts_allowed"]) + 0.5 * (t["pts"] == t["pts_allowed"]), np.nan)
    t["pd"] = t["pts"] - t["pts_allowed"]
    key = t["team"] + t["season"].astype(str)
    g = t.groupby(key, sort=False)
    gp = (played.astype(float).groupby(key, sort=False).cumsum() - played.astype(float))
    wins = t["w"].fillna(0).groupby(key, sort=False).cumsum() - t["w"].fillna(0)
    pdc = t["pd"].fillna(0).groupby(key, sort=False).cumsum() - t["pd"].fillna(0)
    t["rec_gp"] = gp
    t["rec_wpct"] = (wins + 1.5) / (gp + 3)          # shrunk toward .500 early in the season
    t["rec_pd_pg"] = pdc / gp.replace(0, np.nan)
    t["rec_pd_l3"] = g["pd"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    # streak: + consecutive wins / - consecutive losses entering the game
    streaks = []
    for _, s in g["w"]:
        cur, out = 0, []
        for v in s.to_numpy():
            out.append(cur)
            if v == v:
                cur = (cur + 1 if cur >= 0 else 1) if v == 1 else ((cur - 1 if cur <= 0 else -1) if v == 0 else 0)
        streaks.extend(out)
    t["rec_streak"] = pd.Series(streaks, index=t.index)
    return t[["game_id", "team", "rec_gp", "rec_wpct", "rec_pd_pg", "rec_pd_l3", "rec_streak"]]


def _team_homes(tg: pd.DataFrame, ven: pd.DataFrame) -> pd.DataFrame:
    h = tg[tg["is_home"] == 1][["team", "season", "game_id"]].merge(ven, on="game_id")
    return h.groupby(["team", "season"])[["v_lat", "v_lon"]].median().rename(
        columns={"v_lat": "home_lat", "v_lon": "home_lon"}).reset_index()


def add_context(b: pd.DataFrame, tg: pd.DataFrame, sched: pd.DataFrame, online: bool = True, log=print) -> pd.DataFrame:
    b = add_residuals(b)
    # ---- team & opponent record
    R = team_record(tg)
    b = b.merge(R, on=["game_id", "team"], how="left")
    Ro = R.rename(columns={"team": "opp"})
    Ro.columns = [c if c in ("game_id", "opp") else f"opp_{c}" for c in Ro.columns]
    b = b.merge(Ro, on=["game_id", "opp"], how="left")
    b["wpct_diff"] = b["rec_wpct"] - b["opp_rec_wpct"]

    # ---- player vs this franchise, vs former team, vs opposing head coach
    b = pd.concat([b, _prior_split(b, ["player_id", "opp"], "vsopp")], axis=1)
    first = b.groupby(["player_id", "team"])["kickoff"].min().rename("_first_with").reset_index() \
        .rename(columns={"team": "opp"})
    b = b.merge(first, on=["player_id", "opp"], how="left")
    b["vs_former_team"] = (b["_first_with"] < b["kickoff"]).astype(float)
    b = b.drop(columns="_first_with")
    s = sched[["game_id", "home_team", "away_team", "home_coach", "away_coach", "referee", "gametime", "weekday",
               "total_line", "home_score", "away_score"]].copy()
    coach = pd.concat([s[["game_id", "home_team", "home_coach"]].set_axis(["game_id", "team", "coach"], axis=1),
                       s[["game_id", "away_team", "away_coach"]].set_axis(["game_id", "team", "coach"], axis=1)])
    b = b.merge(coach.rename(columns={"team": "opp", "coach": "opp_coach"}), on=["game_id", "opp"], how="left")
    b["opp_coach"] = b["opp_coach"].fillna("?")
    b = pd.concat([b, _prior_split(b, ["player_id", "opp_coach"], "vscoach")], axis=1)
    # own head coach new this season (scheme change)
    c = coach.merge(tg[["game_id", "team", "season", "kickoff"]], on=["game_id", "team"]).sort_values(["team", "kickoff"])
    first_c = c.groupby(["team", "coach"])["season"].transform("min")
    c["hc_new"] = (c["season"] == first_c).astype(float)
    c.loc[c.groupby("team")["season"].transform("min") == c["season"], "hc_new"] = np.nan  # unknown at data start
    b = b.merge(c[["game_id", "team", "hc_new"]], on=["game_id", "team"], how="left")
    b = b.drop(columns="opp_coach")

    # ---- referee crew: game total minus closing total in their previous games
    s = s.merge(tg[["game_id", "kickoff"]].drop_duplicates("game_id"), on="game_id")
    s["tot_res"] = s["home_score"] + s["away_score"] - s["total_line"]
    s = s.sort_values("kickoff")
    s["referee"] = s["referee"].fillna("?")
    gr = s.groupby("referee", sort=False)["tot_res"]
    s["ref_tot_res"] = gr.transform(lambda x: x.shift(1).ewm(halflife=16, min_periods=4).mean())
    s.loc[s["referee"] == "?", "ref_tot_res"] = np.nan
    b = b.merge(s[["game_id", "ref_tot_res"]], on="game_id", how="left")

    # ---- schedule spot
    hour = pd.to_datetime(b["kickoff"]).dt.hour
    dow = pd.to_datetime(b["kickoff"]).dt.dayofweek
    b["primetime"] = (hour >= 19).astype(float)
    b["early_window"] = (hour < 14).astype(float)
    b["thursday"] = (dow == 3).astype(float)
    b["short_week"] = (b["rest"] <= 5).astype(float)
    opp_rest = tg[["game_id", "team", "rest"]].rename(columns={"team": "opp", "rest": "_opp_rest"})
    b = b.merge(opp_rest, on=["game_id", "opp"], how="left")
    b["rest_edge"] = b["rest"] - b["_opp_rest"]
    b = b.drop(columns="_opp_rest")

    # ---- venue, travel, altitude, weather
    wx = weather.game_weather(sched, online=online, log=log)
    b = b.merge(wx, on="game_id", how="left")
    homes = _team_homes(tg, wx[["game_id", "v_lat", "v_lon"]])
    b = b.merge(homes, on=["team", "season"], how="left")
    b["travel_km"] = weather.haversine_km(b["home_lat"], b["home_lon"], b["v_lat"], b["v_lon"])
    b["tz_shift"] = (((b["v_lon"] - b["home_lon"]) + 180) % 360 - 180) / 15.0
    b["body_clock"] = ((b["home_lon"] < -110) & (b["tz_shift"] > 1.5) & (hour < 14)).astype(float)
    b["altitude"] = b["v_elev"]
    # schedule temp/wind are blank for unplayed games; forecast fills them so old features stay usable
    b["temp"] = b["temp"].fillna(b["wx_temp"])
    b["wind"] = b["wind"].fillna(b["wx_wind"])
    return b.drop(columns=["home_lat", "home_lon", "v_lat", "v_lon", "v_elev"])
