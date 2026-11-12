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
PBP_PLAYER_COLS = ["rz_targets", "ez_targets", "deep_targets", "long_rec", "rz_carries", "gl_carries", "long_rush"]
DERIVED = ["td", "rr_yds", "carry_share", "offense_snaps", "offense_pct"]
ROLL = STAT_COLS + PBP_PLAYER_COLS + DERIVED
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
                         rest=s["home_rest"], pts=s["home_score"], pts_allowed=s["away_score"])
    a = s[common].assign(team=s["away_team"], opp=s["home_team"], is_home=0.0, spread=-s["spread_line"],
                         rest=s["away_rest"], pts=s["away_score"], pts_allowed=s["home_score"])
    tg = pd.concat([h, a], ignore_index=True)
    tg.loc[tg["location"] == "Neutral", "is_home"] = 0.5
    tg["implied_total"] = tg["total_line"] / 2 + tg["spread"] / 2
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
                   lr=t["receiving_yards"].where(t["complete_pass"] == 1)) \
        .groupby(["game_id", "receiver_player_id"]).agg(rz_targets=("rz", "sum"), ez_targets=("ez", "sum"),
                                                         deep_targets=("deep", "sum"), long_rec=("lr", "max")) \
        .reset_index().rename(columns={"receiver_player_id": "player_id"})
    r = p[p["rusher_player_id"].notna()]
    rush = r.assign(gl=(r["yardline_100"] <= 5).astype(float)) \
        .groupby(["game_id", "rusher_player_id"]).agg(rz_carries=("rz", "sum"), gl_carries=("gl", "sum"),
                                                       long_rush=("rushing_yards", "max")) \
        .reset_index().rename(columns={"rusher_player_id": "player_id"})
    # --- team offense / defense
    p["db_epa"] = p["epa"].where(p["qb_dropback"] == 1)
    p["ru_epa"] = p["epa"].where(p["qb_dropback"] != 1)
    neutral = p["wp"].between(0.2, 0.8) & (p["half_seconds_remaining"] > 120)
    p["n_pass"] = p["qb_dropback"].where(neutral)
    agg = dict(plays=("epa", "size"), dropbacks=("qb_dropback", "sum"), proe=("pass_oe", "mean"),
               epa_play=("epa", "mean"), success=("success", "mean"), rz_plays=("rz", "sum"),
               db_epa=("db_epa", "mean"), ru_epa=("ru_epa", "mean"), neutral_pass=("n_pass", "mean"),
               sacks=("sack", "sum"), ints=("interception", "sum"))
    off = p.groupby(["game_id", "posteam"]).agg(**agg).reset_index().rename(columns={"posteam": "team"})
    dfn = p.groupby(["game_id", "defteam"]).agg(**agg).reset_index().rename(columns={"defteam": "team"})
    return rec, rush, off, dfn


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
           "implied_total", "rest", "indoors", "turf", "temp", "wind", "div_game", "is_post"]
    b = b.merge(tg[ctx], on=["game_id", "team"], how="inner")
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
                "off_neutral_pass", "off_sacks", "off_ints"]
    def_cols = ["pts_allowed"] + [f"alw_{c}" for c in off_cols if c != "pts"]
    Tf = pd.concat([T[["game_id", "team"]], _lag_ewm(T, "team", off_cols, 4, "e4"),
                    _lag_ewm(T, "team", def_cols, 5, "e5")], axis=1)
    b = b.merge(Tf[["game_id", "team"] + [f"{c}_e4" for c in off_cols]], on=["game_id", "team"], how="left")
    D = Tf[["game_id", "team"] + [f"{c}_e5" for c in def_cols]].rename(columns={"team": "opp"})
    D.columns = [c if c in ("game_id", "opp") else "opp_" + c for c in D.columns]
    b = b.merge(D, on=["game_id", "opp"], how="left")

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
             "ypa_e6", "games_career", "attempts_e2", "passing_yards_e2"]
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
    outcomes = set(STAT_COLS + PBP_PLAYER_COLS + DERIVED + ["td", "rr_yds"])
    drop = outcomes | set(ID_COLS) | {"qb_id"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
