"""Build the player-game modelling frame.

One row per (offensive skill player, game). Every feature is known before kickoff:
rolling form is computed from *prior* games only (shift(1) before any window), and the
upcoming slate is appended as stat-less "future" rows so training and inference share one code path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FIRST_SEASON, SKILL_POS
from .data import current_season

POS_CODE = {"QB": 0, "RB": 1, "FB": 2, "WR": 3, "TE": 4}
POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}

STAT_COLS = [
    "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions", "sacks_suffered",
    "passing_air_yards", "passing_epa", "passing_cpoe", "carries", "rushing_yards", "rushing_tds",
    "rushing_epa", "rushing_first_downs", "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa", "receiving_first_downs",
    "target_share", "air_yards_share", "wopr", "fantasy_points_ppr",
]
PBP_PLAYER_COLS = ["rz_targets", "ez_targets", "deep_targets", "long_rec", "rz_carries", "gl_carries", "long_rush",
                   "xcatch", "expl_rec", "third_targets", "expl_rush", "stuffed", "scrambles", "scramble_yds", "kneels"]
# opportunity-based expectations (ffverse), tracking data (Next Gen Stats) and charting (PFR), one value per player-game
FF_COLS = {"receptions_exp": "xrec", "rec_yards_gained_exp": "xrec_yds", "rec_touchdown_exp": "xrec_td",
           "rush_yards_gained_exp": "xrush_yds", "rush_touchdown_exp": "xrush_td", "pass_yards_gained_exp": "xpass_yds",
           "pass_touchdown_exp": "xpass_td", "pass_completions_exp": "xpass_comp", "pass_interception_exp": "xpass_int",
           "total_first_down_exp": "xfd", "total_fantasy_points_exp": "xfp"}
NGS_COLS = {
    "receiving": {"avg_cushion": "ngs_cushion", "avg_separation": "ngs_sep", "avg_intended_air_yards": "ngs_iay",
                  "percent_share_of_intended_air_yards": "ngs_iay_share", "avg_yac_above_expectation": "ngs_yac_oe"},
    "rushing": {"efficiency": "ngs_ru_eff", "percent_attempts_gte_eight_defenders": "ngs_box8", "avg_time_to_los": "ngs_tlos",
                "rush_yards_over_expected_per_att": "ngs_ryoe_att", "rush_pct_over_expected": "ngs_rpoe"},
    "passing": {"avg_time_to_throw": "ngs_ttt", "avg_intended_air_yards": "ngs_p_iay", "aggressiveness": "ngs_aggr",
                "completion_percentage_above_expectation": "ngs_cpoe", "avg_air_yards_to_sticks": "ngs_ay_sticks",
                "expected_completion_percentage": "ngs_xcomp"},
}
PFR_COLS = {
    "rec": {"receiving_drop": "pfr_drops", "receiving_broken_tackles": "pfr_rec_bt"},
    "rush": {"rushing_yards_before_contact": "pfr_ybc", "rushing_yards_after_contact": "pfr_yac",
             "rushing_broken_tackles": "pfr_ru_bt"},
    "pass": {"passing_bad_throws": "pfr_bad_throws", "times_pressured": "pfr_pressured", "times_blitzed": "pfr_blitzed",
             "passing_drops": "pfr_p_drops"},
}
PFR_DEF = {"def_pressures": "dpress", "def_times_blitzed": "dblitz", "def_times_hurried": "dhurry",
           "def_missed_tackles": "dmiss", "def_sacks": "dsack"}
XF = list(FF_COLS.values())
TRACK = [v for d in list(NGS_COLS.values()) + list(PFR_COLS.values()) for v in d.values()]
EXTRA = XF + TRACK + ["xtd", "xrr_yds"]
# the stat behind each prop: the previous HIST games of these are kept so the ladder model can see how
# often this player actually cleared a given strike
HIST = 12
HIST_COLS = ["passing_yards", "passing_tds", "attempts", "completions", "passing_interceptions", "rushing_yards",
             "carries", "receptions", "receiving_yards", "rr_yds", "td"]
DERIVED = ["td", "rr_yds", "carry_share", "offense_snaps", "offense_pct"]
ROLL = STAT_COLS + PBP_PLAYER_COLS + DERIVED + EXTRA
L1 = ["offense_pct", "targets", "carries", "attempts", "receiving_yards", "rushing_yards", "passing_yards", "td"]
MAIN = ["passing_yards", "attempts", "completions", "passing_tds", "rushing_yards", "carries", "receptions",
        "receiving_yards", "targets", "td", "rr_yds", "offense_pct"]
ID_COLS = ["player_id", "name", "position", "team", "opp", "game_id", "season", "week", "kickoff",
           "is_future", "is_primary_qb"]


# ----------------------------------------------------------------------------- helpers
def _lag_ewm(df: pd.DataFrame, key: str, cols: list[str], hl: float, suffix: str, shift: int = 1) -> pd.DataFrame:
    """Per-`key` exponentially weighted mean of the previous games (df must be time-sorted within key)."""
    src = df.groupby(key, sort=False)[cols].shift(shift) if shift else df[cols]
    out = src.groupby(df[key], sort=False).ewm(halflife=hl, min_periods=1, ignore_na=True).mean()
    out = out.reset_index(level=0, drop=True).reindex(df.index)
    out.columns = [f"{c}_{suffix}" for c in cols]
    return out


def team_games(sched: pd.DataFrame) -> pd.DataFrame:
    s = sched[sched["season"] >= FIRST_SEASON].copy()
    s["kickoff"] = pd.to_datetime(s["gameday"].astype(str) + " " + s["gametime"].fillna("13:00").astype(str),
                                  errors="coerce")
    common = ["game_id", "season", "week", "game_type", "kickoff", "total_line", "roof", "surface", "temp",
              "wind", "div_game", "location"]
    h = s[common].assign(team=s["home_team"], opp=s["away_team"], is_home=1.0, spread=s["spread_line"],
                         rest=s["home_rest"], opp_rest=s["away_rest"], pts=s["home_score"], pts_allowed=s["away_score"])
    a = s[common].assign(team=s["away_team"], opp=s["home_team"], is_home=0.0, spread=-s["spread_line"],
                         rest=s["away_rest"], opp_rest=s["home_rest"], pts=s["away_score"], pts_allowed=s["home_score"])
    tg = pd.concat([h, a], ignore_index=True)
    tg.loc[tg["location"] == "Neutral", "is_home"] = 0.5
    tg["implied_total"] = tg["total_line"] / 2 + tg["spread"] / 2
    tg["opp_implied_total"] = tg["total_line"] - tg["implied_total"]
    tg["abs_spread"] = tg["spread"].abs()
    tg["rest_diff"] = tg["rest"] - tg["opp_rest"]
    tg["kick_hour"] = tg["kickoff"].dt.hour + tg["kickoff"].dt.minute / 60
    tg["week_num"] = tg["week"].astype(float)
    tg["indoors"] = tg["roof"].isin(["dome", "closed"]).astype(float)
    tg["turf"] = np.where(tg["surface"].isna(), np.nan, (~tg["surface"].astype(str).str.contains("grass")).astype(float))
    tg["is_post"] = (tg["game_type"] != "REG").astype(float)
    tg["played"] = tg["pts"].notna()
    tg["wind"] = tg["wind"].where(tg["indoors"] == 0, 0.0)
    return tg.sort_values(["team", "kickoff"]).reset_index(drop=True)


def _pbp_aggregates(pbp: pd.DataFrame):
    p = pbp[(pbp["two_point_attempt"] != 1) & ((pbp["pass"] == 1) | (pbp["rush"] == 1))].copy()
    p["rz"] = (p["yardline_100"] <= 20).astype(float)
    # --- player level
    t = p[p["receiver_player_id"].notna() & (p["sack"] != 1)]
    rec = t.assign(ez=(t["air_yards"] >= t["yardline_100"]).astype(float), deep=(t["air_yards"] >= 20).astype(float),
                   lr=t["receiving_yards"].where(t["complete_pass"] == 1),
                   xr=(t["receiving_yards"] >= 20).astype(float), d3=(t["down"] >= 3).astype(float)) \
        .groupby(["game_id", "receiver_player_id"]).agg(rz_targets=("rz", "sum"), ez_targets=("ez", "sum"),
                                                         deep_targets=("deep", "sum"), long_rec=("lr", "max"),
                                                         xcatch=("cp", "sum"), expl_rec=("xr", "sum"),
                                                         third_targets=("d3", "sum")) \
        .reset_index().rename(columns={"receiver_player_id": "player_id"})
    r = p[p["rusher_player_id"].notna()]
    rush = r.assign(gl=(r["yardline_100"] <= 5).astype(float), xr=(r["rushing_yards"] >= 10).astype(float),
                    st=(r["rushing_yards"] <= 0).astype(float), sc=(r["qb_scramble"] == 1).astype(float),
                    scy=r["rushing_yards"].where(r["qb_scramble"] == 1, 0.0)) \
        .groupby(["game_id", "rusher_player_id"]).agg(rz_carries=("rz", "sum"), gl_carries=("gl", "sum"),
                                                       long_rush=("rushing_yards", "max"), expl_rush=("xr", "sum"),
                                                       stuffed=("st", "sum"), scrambles=("sc", "sum"),
                                                       scramble_yds=("scy", "sum")) \
        .reset_index().rename(columns={"rusher_player_id": "player_id"})
    kn = pbp[(pbp["qb_kneel"] == 1) & pbp["rusher_player_id"].notna()].groupby(["game_id", "rusher_player_id"]).size() \
        .rename("kneels").reset_index().rename(columns={"rusher_player_id": "player_id"})
    rush = rush.merge(kn, on=["game_id", "player_id"], how="outer")
    # --- team offense / defense
    db = p["qb_dropback"] == 1
    p["db_epa"] = p["epa"].where(db)
    p["ru_epa"] = p["epa"].where(~db)
    neutral = p["wp"].between(0.2, 0.8) & (p["half_seconds_remaining"] > 120)
    p["n_pass"] = p["qb_dropback"].where(neutral)
    p["air"] = p["air_yards"].where((p["pass"] == 1) & (p["sack"] != 1))
    p["expl"] = np.where(db, p["yards_gained"] >= 20, p["yards_gained"] >= 10).astype(float)
    p["hit"] = p["qb_hit"].where(db)
    p["ru_yds"] = p["yards_gained"].where(~db)
    p["db_yds"] = p["yards_gained"].where(db)
    p["stuff"] = (p["yards_gained"] <= 0).astype(float).where(~db)
    p["rz_pass"] = p["qb_dropback"].where(p["rz"] == 1)
    agg = dict(plays=("epa", "size"), dropbacks=("qb_dropback", "sum"), proe=("pass_oe", "mean"),
               epa_play=("epa", "mean"), success=("success", "mean"), rz_plays=("rz", "sum"),
               db_epa=("db_epa", "mean"), ru_epa=("ru_epa", "mean"), neutral_pass=("n_pass", "mean"),
               sacks=("sack", "sum"), ints=("interception", "sum"),
               shotgun=("shotgun", "mean"), no_huddle=("no_huddle", "mean"), adot=("air", "mean"),
               yac=("yards_after_catch", "mean"), explosive=("expl", "mean"), qb_hit=("hit", "mean"),
               ypc=("ru_yds", "mean"), ypd=("db_yds", "mean"), stuff=("stuff", "mean"), cp=("cp", "mean"),
               fd=("first_down", "mean"), rz_pass=("rz_pass", "mean"))
    off = p.groupby(["game_id", "posteam"]).agg(**agg).reset_index().rename(columns={"posteam": "team"})
    dfn = p.groupby(["game_id", "defteam"]).agg(**agg).reset_index().rename(columns={"defteam": "team"})
    return rec, rush, off, dfn


def _extra_player_stats(raw: dict, b: pd.DataFrame) -> pd.DataFrame:
    """Join opportunity / tracking / charting data (one value per player-game) onto the base frame."""
    ff = raw.get("ff_opportunity")
    for c in XF:
        b[c] = np.nan
    if ff is not None and len(ff):
        ff = ff.rename(columns=FF_COLS).dropna(subset=["player_id", "game_id"]).drop_duplicates(["player_id", "game_id"])
        b = b.drop(columns=XF).merge(ff[["player_id", "game_id"] + XF], on=["player_id", "game_id"], how="left")
        # a player with no opportunities has no row; that is a true zero, but only for games the feed covers
        covered = b["game_id"].isin(ff["game_id"]) & ~b["is_future"]
        b.loc[covered, XF] = b.loc[covered, XF].fillna(0.0)
    b["xtd"] = b["xrec_td"] + b["xrush_td"]
    b["xrr_yds"] = b["xrec_yds"] + b["xrush_yds"]
    for kind, cols in NGS_COLS.items():
        d = raw.get(f"ngs_{kind}")
        if d is None or not len(d):
            for v in cols.values():
                b[v] = np.nan
            continue
        d = d.copy()
        # Next Gen numbers the Super Bowl one week later than nflverse does
        last_post = d[d["season_type"] == "POST"].groupby("season")["week"].transform("max")
        sb = (d["season_type"] == "POST") & (d["week"] == last_post.reindex(d.index))
        d.loc[sb, "week"] -= 1
        d = d.rename(columns=dict(cols, player_gsis_id="player_id"))[["season", "week", "player_id"] + list(cols.values())] \
            .dropna(subset=["player_id"]).drop_duplicates(["season", "week", "player_id"])
        d["season"] = d["season"].astype(b["season"].dtype); d["week"] = d["week"].astype(b["week"].dtype)
        b = b.merge(d, on=["season", "week", "player_id"], how="left")
    ids = raw["players"][["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    for kind, cols in PFR_COLS.items():
        d = raw.get(f"pfr_{kind}")
        if d is None or not len(d):
            for v in cols.values():
                b[v] = np.nan
            continue
        d = d.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner").rename(columns=dict(cols, gsis_id="player_id"))
        d = d[["game_id", "player_id"] + list(cols.values())].drop_duplicates(["game_id", "player_id"])
        b = b.merge(d, on=["game_id", "player_id"], how="left")
    b.loc[b["is_future"], EXTRA] = np.nan
    return b


# ----------------------------------------------------------------------------- base frame
def _base(raw: dict, tg: pd.DataFrame) -> pd.DataFrame:
    ps = raw["player_stats"]
    ps = ps[ps["position"].isin(SKILL_POS)].copy()
    for c in ("special_teams_tds", "def_tds", "fumble_recovery_tds"):
        if c not in ps.columns:
            ps[c] = 0.0
    ps["td"] = ps[["rushing_tds", "receiving_tds", "special_teams_tds", "def_tds", "fumble_recovery_tds"]].fillna(0).sum(axis=1)
    keep = ["player_id", "game_id", "team", "position", "td"] + STAT_COLS
    ps = ps[keep].drop_duplicates(["player_id", "game_id"])

    ids = raw["players"][["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    sn = raw["snap_counts"]
    sn = sn[(sn["offense_snaps"] > 0) & sn["position"].isin(SKILL_POS)]
    sn = sn.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner") \
        .rename(columns={"gsis_id": "player_id"})[["player_id", "game_id", "team", "position", "offense_snaps", "offense_pct"]] \
        .drop_duplicates(["player_id", "game_id"])

    b = ps.merge(sn, on=["player_id", "game_id"], how="outer", suffixes=("", "_sn"))
    b["team"] = b["team"].fillna(b["team_sn"])
    b["position"] = b["position"].fillna(b["position_sn"])
    involved = b[["attempts", "carries", "targets"]].fillna(0).sum(axis=1) > 0
    b = b[b["offense_snaps"].notna() | involved].drop(columns=["team_sn", "position_sn"])
    zero = [c for c in STAT_COLS + ["td"] if c != "passing_cpoe"]
    b[zero] = b[zero].fillna(0.0)
    return b


def _future_rows(raw: dict, tg: pd.DataFrame, now: pd.Timestamp, horizon_days: int = 8) -> pd.DataFrame:
    up = tg[(~tg["played"]) & (tg["kickoff"] >= now - pd.Timedelta(hours=1)) &
            (tg["kickoff"] <= now + pd.Timedelta(days=horizon_days))]
    if up.empty:
        return pd.DataFrame()
    ros = raw["rosters_weekly"]
    ros = ros[ros["position"].isin(SKILL_POS) & ros["gsis_id"].notna()]
    latest = ros[ros["week"] == ros.groupby("team")["week"].transform("max")]
    act = latest[latest["status"] == "ACT"][["gsis_id", "team", "position"]].drop_duplicates("gsis_id") \
        .rename(columns={"gsis_id": "player_id"})
    inj = raw["injuries"]
    cur = current_season()
    weeks = up["week"].unique()
    out = inj[(inj["season"] == cur) & inj["week"].isin(weeks) & inj["report_status"].isin(["Out", "Doubtful"])]
    act = act[~act["player_id"].isin(out["gsis_id"])]
    f = act.merge(up[["team", "game_id"]], on="team", how="inner")
    f["is_future"] = True
    return f


# ----------------------------------------------------------------------------- main entry
def build(raw: dict, include_future: bool = True, qb_override: dict | None = None,
          now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return the full modelling frame. `qb_override` maps team -> player_id of the expected
    starting QB for upcoming games (the scanner derives it from which QB Kalshi lists)."""
    now = now or pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    tg = team_games(raw["schedules"])
    rec, rush, off, dfn = _pbp_aggregates(raw["pbp"])

    b = _base(raw, tg)
    b["is_future"] = False
    if include_future:
        fut = _future_rows(raw, tg, now)
        if len(fut):
            b = pd.concat([b, fut], ignore_index=True)
    b["is_future"] = b["is_future"].astype(bool)
    b = b.merge(rec, on=["game_id", "player_id"], how="left").merge(rush, on=["game_id", "player_id"], how="left")
    hist = ~b["is_future"]
    b.loc[hist, PBP_PLAYER_COLS] = b.loc[hist, PBP_PLAYER_COLS].fillna(0.0)

    ctx = ["game_id", "team", "opp", "season", "week", "kickoff", "is_home", "spread", "total_line",
           "implied_total", "rest", "indoors", "turf", "temp", "wind", "div_game", "is_post",
           "opp_implied_total", "abs_spread", "opp_rest", "rest_diff", "kick_hour", "week_num"]
    b = b.merge(tg[ctx], on=["game_id", "team"], how="inner")
    b = _extra_player_stats(raw, b)
    b["posg"] = b["position"].map(POS_GROUP)
    b["rr_yds"] = b["rushing_yards"] + b["receiving_yards"]
    team_carries = b.groupby(["game_id", "team"])["carries"].transform("sum")
    b["carry_share"] = np.where(team_carries > 0, b["carries"] / team_carries, 0.0)
    b.loc[b["is_future"], "carry_share"] = np.nan

    b = b.sort_values(["player_id", "kickoff"]).reset_index(drop=True)

    # ---- player rolling form
    g = b.groupby("player_id", sort=False)
    feats = [_lag_ewm(b, "player_id", ROLL, 2, "e2"), _lag_ewm(b, "player_id", ROLL, 6, "e6")]
    sh = g[list(dict.fromkeys(MAIN + L1))].shift(1)
    l1 = sh[L1].groupby(b["player_id"], sort=False).ffill()
    l1.columns = [f"{c}_l1" for c in L1]
    sd = sh[MAIN].groupby(b["player_id"], sort=False).rolling(8, min_periods=3).std() \
        .reset_index(level=0, drop=True).reindex(b.index)
    sd.columns = [f"{c}_sd8" for c in MAIN]
    car = sh[MAIN].groupby(b["player_id"], sort=False).expanding(min_periods=1).mean() \
        .reset_index(level=0, drop=True).reindex(b.index)
    car.columns = [f"{c}_car" for c in MAIN]
    key_ps = b["player_id"].astype(str) + "_" + b["season"].astype(str)
    shs = b.groupby(key_ps, sort=False)[MAIN].shift(1)
    szn = shs.groupby(key_ps, sort=False).expanding(min_periods=1).mean().reset_index(level=0, drop=True).reindex(b.index)
    szn.columns = [f"{c}_szn" for c in MAIN]
    b = pd.concat([b] + feats + [l1, sd, car, szn], axis=1)
    b["games_career"] = g.cumcount().astype(float)
    b["games_season"] = b.groupby(key_ps, sort=False).cumcount().astype(float)
    b["days_since_last"] = (b["kickoff"] - g["kickoff"].shift(1)).dt.days.clip(upper=400)
    b["team_changed"] = (b["team"] != g["team"].shift(1)).astype(float).where(b["games_career"] > 0)
    hs = [g[HIST_COLS].shift(j).astype(np.float32).add_suffix(f"_h{j}") for j in range(1, HIST + 1)]
    b = pd.concat([b] + hs, axis=1)
    g = b.groupby("player_id", sort=False)
    nz = lambda c: b[c].replace(0, np.nan)
    b["rec_yds_oe_e6"] = b["receiving_yards_e6"] - b["xrec_yds_e6"]
    b["rush_yds_oe_e6"] = b["rushing_yards_e6"] - b["xrush_yds_e6"]
    b["rec_oe_e6"] = b["receptions_e6"] - b["xrec_e6"]
    b["td_oe_e6"] = b["td_e6"] - b["xtd_e6"]
    b["pass_yds_oe_e6"] = b["passing_yards_e6"] - b["xpass_yds_e6"]
    b["xcatch_rate_e6"] = b["xcatch_e6"] / nz("targets_e6")
    b["drop_rate_e6"] = b["pfr_drops_e6"] / nz("targets_e6")
    b["ybc_att_e6"] = b["pfr_ybc_e6"] / nz("carries_e6")
    b["yac_att_e6"] = b["pfr_yac_e6"] / nz("carries_e6")
    b["stuff_rate_e6"] = b["stuffed_e6"] / nz("carries_e6")
    b["expl_rush_rate_e6"] = b["expl_rush_e6"] / nz("carries_e6")
    b["expl_rec_rate_e6"] = b["expl_rec_e6"] / nz("targets_e6")
    b["pressure_rate_e6"] = b["pfr_pressured_e6"] / (b["attempts_e6"] + b["sacks_suffered_e6"]).replace(0, np.nan)
    b["bad_throw_rate_e6"] = b["pfr_bad_throws_e6"] / nz("attempts_e6")
    b["scramble_share_e6"] = b["scrambles_e6"] / nz("carries_e6")
    b["tgt_per_snap_e6"] = b["targets_e6"] / nz("offense_snaps_e6")
    b["carry_per_snap_e6"] = b["carries_e6"] / nz("offense_snaps_e6")
    b["comp_pct_e6"] = b["completions_e6"] / nz("attempts_e6")
    b["td_per_att_e6"] = b["passing_tds_e6"] / nz("attempts_e6")
    b["int_per_att_e6"] = b["passing_interceptions_e6"] / nz("attempts_e6")
    b["sack_rate_e6"] = b["sacks_suffered_e6"] / (b["attempts_e6"] + b["sacks_suffered_e6"]).replace(0, np.nan)
    b["adot_e6"] = b["receiving_air_yards_e6"] / b["targets_e6"].replace(0, np.nan)
    b["ypc_e6"] = b["rushing_yards_e6"] / b["carries_e6"].replace(0, np.nan)
    b["ypt_e6"] = b["receiving_yards_e6"] / b["targets_e6"].replace(0, np.nan)
    b["catch_rate_e6"] = b["receptions_e6"] / b["targets_e6"].replace(0, np.nan)
    b["ypa_e6"] = b["passing_yards_e6"] / b["attempts_e6"].replace(0, np.nan)

    # ---- team offense & opponent defense form
    tsum = b[~b["is_future"]].groupby(["game_id", "team"]).agg(
        tm_att=("attempts", "sum"), tm_carries=("carries", "sum"), tm_pass_yds=("passing_yards", "sum"),
        tm_rush_yds=("rushing_yards", "sum"), tm_pass_tds=("passing_tds", "sum"), tm_rush_tds=("rushing_tds", "sum"),
        tm_td=("td", "sum")).reset_index()
    T = tg[["game_id", "team", "opp", "kickoff", "pts", "pts_allowed"]].merge(tsum, on=["game_id", "team"], how="left") \
        .merge(off.add_prefix("off_").rename(columns={"off_game_id": "game_id", "off_team": "team"}),
               on=["game_id", "team"], how="left")
    # what this team's defense allowed = the opponent offense's line in the same game
    opp_view = T.drop(columns=["opp", "kickoff", "pts", "pts_allowed"]).rename(columns={"team": "opp"})
    opp_view.columns = [c if c in ("game_id", "opp") else f"alw_{c}" for c in opp_view.columns]
    T = T.merge(opp_view, on=["game_id", "opp"], how="left").sort_values(["team", "kickoff"]).reset_index(drop=True)
    off_cols = ["pts", "tm_att", "tm_carries", "tm_pass_yds", "tm_rush_yds", "tm_td", "off_plays", "off_dropbacks",
                "off_proe", "off_epa_play", "off_success", "off_rz_plays", "off_db_epa", "off_ru_epa",
                "off_neutral_pass", "off_sacks", "off_ints", "off_shotgun", "off_no_huddle", "off_adot", "off_yac",
                "off_explosive", "off_qb_hit", "off_ypc", "off_ypd", "off_stuff", "off_cp", "off_fd", "off_rz_pass"]
    def_cols = ["pts_allowed"] + [f"alw_{c}" for c in off_cols if c not in ("pts", "off_shotgun", "off_no_huddle")]
    Tf = pd.concat([T[["game_id", "team"]], _lag_ewm(T, "team", off_cols, 4, "e4"),
                    _lag_ewm(T, "team", def_cols, 5, "e5")], axis=1)
    b = b.merge(Tf[["game_id", "team"] + [f"{c}_e4" for c in off_cols]], on=["game_id", "team"], how="left")
    D = Tf[["game_id", "team"] + [f"{c}_e5" for c in def_cols]].rename(columns={"team": "opp"})
    D.columns = [c if c in ("game_id", "opp") else "opp_" + c for c in D.columns]
    b = b.merge(D, on=["game_id", "opp"], how="left")
    # the opponent's own offense shapes the game script (their pace and scoring decide how often we have the ball)
    O = Tf[["game_id", "team"] + [f"{c}_e4" for c in ("pts", "off_plays", "off_epa_play", "off_neutral_pass", "off_success")]] \
        .rename(columns={"team": "opp"})
    O.columns = [c if c in ("game_id", "opp") else "oppoff_" + c for c in O.columns]
    b = b.merge(O, on=["game_id", "opp"], how="left")
    # charted pass rush / tackling of the opposing defense (PFR, 2018+)
    pd_ = raw.get("pfr_def")
    if pd_ is not None and len(pd_):
        dd = pd_.rename(columns=PFR_DEF).groupby(["game_id", "team"])[list(PFR_DEF.values())].sum(min_count=1).reset_index()
        dd = tg[["game_id", "team", "kickoff"]].merge(dd, on=["game_id", "team"], how="left").sort_values(["team", "kickoff"]) \
            .reset_index(drop=True)
        df_ = pd.concat([dd[["game_id", "team"]], _lag_ewm(dd, "team", list(PFR_DEF.values()), 5, "e5")], axis=1) \
            .rename(columns={"team": "opp"})
        df_.columns = [c if c in ("game_id", "opp") else "opp_" + c for c in df_.columns]
        b = b.merge(df_, on=["game_id", "opp"], how="left")

    # ---- how the team's work is split among the players who are actually available
    for share, vol, out in (("target_share_e6", "tm_att_e4", "targets"), ("carry_share_e6", "tm_carries_e4", "carries")):
        sh_ = b[share].fillna(0.0)
        tot = sh_.groupby([b["game_id"], b["team"]]).transform("sum")
        b[f"{share}_norm"] = sh_ / tot.replace(0, np.nan)
        b[f"{share}_present"] = tot
        b[f"exp_{out}"] = b[f"{share}_norm"] * b[vol]
    b["snap_rank_pos"] = b.groupby(["game_id", "team", "posg"])["offense_pct_e6"].rank(ascending=False, method="min")
    b["tgt_rank_team"] = b.groupby(["game_id", "team"])["targets_e6"].rank(ascending=False, method="min")
    b["carry_rank_team"] = b.groupby(["game_id", "team"])["carries_e6"].rank(ascending=False, method="min")
    b["n_pos_present"] = b.groupby(["game_id", "team", "posg"])["player_id"].transform("size").astype(float)

    # ---- what the opponent allows to this position group
    pa = b[~b["is_future"]].groupby(["game_id", "opp", "posg"]).agg(
        pa_rec_yds=("receiving_yards", "sum"), pa_rec=("receptions", "sum"), pa_targets=("targets", "sum"),
        pa_rush_yds=("rushing_yards", "sum"), pa_carries=("carries", "sum"), pa_td=("td", "sum")).reset_index()
    P = tg[["game_id", "team", "kickoff", "played"]].rename(columns={"team": "opp"}).merge(
        pd.DataFrame({"posg": ["QB", "RB", "WR", "TE"]}), how="cross").merge(pa, on=["game_id", "opp", "posg"], how="left")
    P["k"] = P["opp"] + "_" + P["posg"]
    P = P.sort_values(["k", "kickoff"]).reset_index(drop=True)
    pa_cols = ["pa_rec_yds", "pa_rec", "pa_targets", "pa_rush_yds", "pa_carries", "pa_td"]
    P.loc[P["played"], pa_cols] = P.loc[P["played"], pa_cols].fillna(0.0)
    Pf = pd.concat([P[["game_id", "opp", "posg"]], _lag_ewm(P, "k", pa_cols, 5, "e5")], axis=1)
    b = b.merge(Pf, on=["game_id", "opp", "posg"], how="left")

    # ---- starting QB context
    b["is_primary_qb"] = False
    qb_rows = b[b["position"] == "QB"]
    h = qb_rows[~qb_rows["is_future"] & (qb_rows["attempts"] > 0)]
    idx_h = h.groupby(["game_id", "team"])["attempts"].idxmax()
    # upcoming games: default to whoever started the team's last game, else the most-used active QB
    last_start = b.loc[idx_h].sort_values("kickoff").groupby("team")["player_id"].last()
    f = qb_rows[qb_rows["is_future"]]
    score = f["attempts_e2"].fillna(-1) + 1000.0 * (f["player_id"] == f["team"].map(last_start))
    idx_f = f.assign(_a=score).groupby(["game_id", "team"])["_a"].idxmax() if len(f) else pd.Series(dtype=int)
    b.loc[list(idx_h) + list(idx_f), "is_primary_qb"] = True
    if qb_override:
        for team, pid in qb_override.items():
            m = b["is_future"] & (b["team"] == team) & (b["position"] == "QB")
            if (m & (b["player_id"] == pid)).any():
                b.loc[m, "is_primary_qb"] = b.loc[m, "player_id"] == pid
    qcols = ["attempts_e6", "passing_yards_e6", "passing_tds_e6", "passing_epa_e6", "passing_cpoe_e6",
             "ypa_e6", "games_career", "attempts_e2", "passing_yards_e2", "xpass_yds_e6", "xpass_td_e6",
             "ngs_p_iay_e6", "ngs_ttt_e6", "ngs_aggr_e6", "scrambles_e6", "comp_pct_e6", "sack_rate_e6",
             "pressure_rate_e6", "bad_throw_rate_e6", "passing_air_yards_e6"]
    Q = b.loc[b["is_primary_qb"], ["game_id", "team", "player_id"] + qcols].drop_duplicates(["game_id", "team"])
    Q.columns = ["game_id", "team", "qb_id"] + [f"qb_{c}" for c in qcols]
    b = b.merge(Q, on=["game_id", "team"], how="left")
    Qo = tg[["game_id", "team", "kickoff"]].merge(Q[["game_id", "team", "qb_id"]], on=["game_id", "team"], how="left") \
        .sort_values(["team", "kickoff"])
    Qo["qb_changed"] = (Qo["qb_id"] != Qo.groupby("team")["qb_id"].shift(1)).astype(float)
    b = b.merge(Qo[["game_id", "team", "qb_changed"]], on=["game_id", "team"], how="left")

    # ---- usage vacated by teammates who played in the last two team games but are absent now
    post = _lag_ewm(b.sort_values(["player_id", "kickoff"]), "player_id", ["target_share", "carry_share"], 6, "post", shift=0)
    b = b.join(post)
    order = tg[["team", "game_id", "kickoff"]].sort_values(["team", "kickoff"]).copy()
    present = b[["team", "game_id", "player_id"]].assign(here=1)
    cand = []
    for lag in (1, 2):
        order[f"prev"] = order.groupby("team")["game_id"].shift(lag)
        c = order[["team", "game_id", "prev"]].dropna().merge(
            b[["team", "game_id", "player_id", "posg", "target_share_post", "carry_share_post"]]
            .rename(columns={"game_id": "prev"}), on=["team", "prev"], how="inner")
        c["lag"] = lag
        cand.append(c)
    cand = pd.concat(cand).sort_values("lag").drop_duplicates(["team", "game_id", "player_id"])
    cand = cand.merge(present, on=["team", "game_id", "player_id"], how="left")
    miss = cand[cand["here"].isna()]
    vac = miss.groupby(["team", "game_id"]).agg(vac_tgt=("target_share_post", "sum"),
                                                vac_carry=("carry_share_post", "sum")).reset_index()
    for pg in ("WR", "TE", "RB"):
        v = miss[miss["posg"] == pg].groupby(["team", "game_id"])["target_share_post"].sum() \
            .rename(f"vac_tgt_{pg.lower()}").reset_index()
        vac = vac.merge(v, on=["team", "game_id"], how="left")
    b = b.merge(vac, on=["team", "game_id"], how="left")
    vcols = ["vac_tgt", "vac_carry", "vac_tgt_wr", "vac_tgt_te", "vac_tgt_rb"]
    b[vcols] = b[vcols].fillna(0.0)
    b = b.drop(columns=["target_share_post", "carry_share_post"])

    # ---- injury report + static bio
    inj = raw["injuries"][["season", "week", "gsis_id", "report_status", "practice_status"]].dropna(subset=["gsis_id"]) \
        .drop_duplicates(["season", "week", "gsis_id"]).rename(columns={"gsis_id": "player_id"})
    inj["season"] = inj["season"].astype(int); inj["week"] = inj["week"].astype(int)
    b = b.merge(inj, on=["season", "week", "player_id"], how="left")
    b["inj_listed"] = b["practice_status"].notna().astype(float)
    b["inj_questionable"] = (b["report_status"] == "Questionable").astype(float)
    prac = b["practice_status"].astype(str)
    b["inj_practice"] = np.select([prac.str.startswith("Did Not"), prac.str.startswith("Limited"), prac.str.startswith("Full")],
                                  [2.0, 1.0, 0.0], default=np.nan)
    pl_ = raw["players"][["gsis_id", "display_name", "birth_date", "rookie_season", "draft_pick", "height", "weight"]] \
        .drop_duplicates("gsis_id").rename(columns={"gsis_id": "player_id", "display_name": "name"})
    b = b.merge(pl_, on="player_id", how="left")
    b["age"] = (b["kickoff"] - pd.to_datetime(b["birth_date"], errors="coerce")).dt.days / 365.25
    b["exp"] = b["season"] - b["rookie_season"]
    b["pos_code"] = b["position"].map(POS_CODE).astype(float)
    b = b.drop(columns=["birth_date", "rookie_season", "report_status", "practice_status", "posg"])
    return b.sort_values(["kickoff", "game_id", "team", "player_id"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything that is known pre-game (i.e. not an identifier and not a same-game outcome)."""
    outcomes = set(STAT_COLS + PBP_PLAYER_COLS + DERIVED + EXTRA + ["td", "rr_yds"])
    drop = outcomes | set(ID_COLS) | {"qb_id"} | set(history_columns(df))
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def history_columns(df: pd.DataFrame, col: str | None = None) -> list[str]:
    """Raw lagged outcomes (`<stat>_h1..hN`): inputs to the per-strike hit rate, never model features directly."""
    cols = [col] if col else HIST_COLS
    return [f"{c}_h{j}" for c in cols for j in range(1, HIST + 1) if f"{c}_h{j}" in df.columns]
