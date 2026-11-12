"""Who is on the other side of the ball, how good they are, and how this player has done against them.

Sources (nflverse, free): snap counts for every defender and offensive lineman (2012+), Pro Football
Reference advanced defense per game (targets / yards / TDs / passer rating allowed in coverage,
pressures, sacks, missed tackles; 2018+), the weekly injury report and the current roster.

Features produced
  per (game, defense):  coverage quality of the defenders expected to play, weighted by expected snap share
                        (CB unit, safeties, linebackers, the top corner), pass-rush pressure rate, and the
                        snap share of regular defenders who are missing (injured / inactive / gone).
  per (game, offense):  offensive-line snap share missing.
  per (player, game):   the player's history against the *specific* defenders he is about to face: his
                        average over/under-performance (actual - his own pre-game rolling average) in
                        previous games against each of them, shrunk toward 0 and weighted by how much
                        each defender plays. Follows defenders across teams, unlike a plain vs-team split.

Free data does not say which corner covered which receiver on a given play, so "vs defender" means
"in games where that defender was on the field for at least ~30% of snaps".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = {"CB": "CB", "S": "S", "FS": "S", "SS": "S", "DB": "S", "LB": "LB", "ILB": "LB", "OLB": "LB",
         "MLB": "LB", "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
         "T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL"}
PFR = ["def_targets", "def_completions_allowed", "def_yards_allowed", "def_receiving_td_allowed",
       "def_ints", "def_pressures", "def_sacks", "def_missed_tackles", "def_tackles_combined"]
RES = {"receiving_yards": "rec_yds", "receptions": "rec", "rushing_yards": "rush_yds", "passing_yards": "pass_yds"}
LG_YPT, LG_CMP, K_TGT = 7.0, 0.64, 12.0  # league-average coverage numbers and shrinkage (in targets)


def _players_map(raw: dict) -> pd.DataFrame:
    p = raw["players"][["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
    return p.drop_duplicates("pfr_id")


def _snaps(raw: dict, tg: pd.DataFrame) -> pd.DataFrame:
    sn = raw["snap_counts"].copy()
    sn["grp"] = sn["position"].map(GROUP)
    sn = sn[sn["grp"].notna()]
    sn["share"] = np.where(sn["grp"] == "OL", sn["offense_pct"], sn["defense_pct"]).astype(float)
    sn = sn[sn["share"] > 0][["pfr_player_id", "game_id", "team", "grp", "share"]] \
        .drop_duplicates(["pfr_player_id", "game_id"]).rename(columns={"pfr_player_id": "pid"})
    sn = sn.merge(tg[["game_id", "team", "kickoff"]], on=["game_id", "team"], how="inner")
    pf = raw.get("pfr_def")
    if pf is not None and len(pf):
        pf = pf[["pfr_player_id", "game_id"] + [c for c in PFR if c in pf.columns]] \
            .rename(columns={"pfr_player_id": "pid"}).drop_duplicates(["pid", "game_id"])
        for c in PFR:
            if c not in pf.columns:
                pf[c] = np.nan
        sn = sn.merge(pf, on=["pid", "game_id"], how="left")
    else:
        for c in PFR:
            sn[c] = np.nan
    return sn.sort_values(["pid", "kickoff"]).reset_index(drop=True)


def _defender_form(sn: pd.DataFrame) -> pd.DataFrame:
    """Per defender-game: form *including* that game (`post`). Prior form = previous row's post."""
    g = sn.groupby("pid", sort=False)
    sn["share_post"] = g["share"].transform(lambda s: s.ewm(halflife=4, min_periods=1).mean())
    roll = g[PFR].rolling(16, min_periods=1).sum().reset_index(level=0, drop=True).reindex(sn.index)
    has = sn["def_targets"].notna()
    n = has.groupby(sn["pid"], sort=False).rolling(16, min_periods=1).sum().reset_index(level=0, drop=True).reindex(sn.index)
    t = roll["def_targets"]
    sn["ypt_post"] = ((roll["def_yards_allowed"] + LG_YPT * K_TGT) / (t + K_TGT)).where(n > 0)
    sn["cmp_post"] = ((roll["def_completions_allowed"] + LG_CMP * K_TGT) / (t + K_TGT)).where(n > 0)
    sn["tdr_post"] = ((roll["def_receiving_td_allowed"] + 0.045 * K_TGT) / (t + K_TGT)).where(n > 0)
    sn["tgt_pg_post"] = (t / n).where(n > 0)
    sn["press_pg_post"] = (roll["def_pressures"] / n).where(n > 0)
    sn["mtk_rate_post"] = (roll["def_missed_tackles"] / (roll["def_tackles_combined"] + roll["def_missed_tackles"] + 5)).where(n > 0)
    return sn


POST = ["share_post", "ypt_post", "cmp_post", "tdr_post", "tgt_pg_post", "press_pg_post", "mtk_rate_post"]


def _lineups(sn: pd.DataFrame, raw: dict, tg: pd.DataFrame, future_games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Rows (game_id, team, pid, grp, w, <prior form>) for every unit that will/did take the field.
    Historical: who actually played, weighted by expected (prior) share. Future: active roster minus
    Out/Doubtful, weighted by each player's latest share."""
    prior = sn.groupby("pid", sort=False)[POST].shift(1)
    prior.columns = [c.replace("_post", "") for c in POST]
    hist = pd.concat([sn[["game_id", "team", "pid", "grp", "kickoff"]], prior], axis=1)
    hist["w"] = hist["share"].fillna(0.35).clip(0.05, 1.0)  # expected share; debut games get a typical value
    hist["q"] = 0.0
    if future_games is None or future_games.empty:
        return hist
    last = sn.groupby("pid", sort=False).tail(1).set_index("pid")
    ids = _players_map(raw)
    ros = raw.get("rosters_weekly", pd.DataFrame())
    if ros.empty:
        return hist
    ros = ros[ros["season"] == season] if "season" in ros.columns else ros
    ros = ros[ros["week"] == ros.groupby("team")["week"].transform("max")]
    ros = ros[ros["status"].isin(["ACT"])]
    pid_col = ros["pfr_id"] if "pfr_id" in ros.columns else pd.Series(np.nan, index=ros.index)
    ros = ros.assign(pid=pid_col.fillna(ros["gsis_id"].map(ids.set_index("gsis_id")["pfr_id"])))
    inj = raw["injuries"]
    weeks = future_games["week"].unique()
    wk = inj[(inj["season"] == season) & inj["week"].isin(weeks)]
    out_ids = set(wk.loc[wk["report_status"].isin(["Out", "Doubtful"]), "gsis_id"].dropna())
    q_ids = set(wk.loc[wk["report_status"].eq("Questionable"), "gsis_id"].dropna())
    ros = ros[~ros["gsis_id"].isin(out_ids) & ros["pid"].isin(last.index)]
    ros = ros.assign(q=ros["gsis_id"].isin(q_ids).astype(float))[["team", "pid", "q"]].drop_duplicates("pid")
    fut = future_games[["game_id", "team", "kickoff"]].drop_duplicates(["game_id", "team"]).merge(ros, on="team")
    L = last.loc[fut["pid"], ["grp"] + POST].reset_index(drop=True)
    L.columns = ["grp"] + [c.replace("_post", "") for c in POST]
    fut = pd.concat([fut.reset_index(drop=True), L], axis=1)
    fut = fut[fut["share"] >= 0.08]
    # history knows game-day inactives; upcoming games only know the injury report, so a Questionable
    # player counts half: half his expected snaps, and half "missing" in unit_features
    fut["w"] = fut["share"].clip(0.05, 1.0) * (1 - 0.5 * fut["q"])
    return pd.concat([hist, fut[hist.columns]], ignore_index=True)


def _wmean(d: pd.DataFrame, col: str, keys: list[str]) -> pd.Series:
    v = d[col]
    w = d["w"].where(v.notna(), 0.0)
    num = (v.fillna(0) * w).groupby([d[k] for k in keys]).sum()
    den = w.groupby([d[k] for k in keys]).sum()
    return (num / den.replace(0, np.nan))


def unit_features(lu: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team-on-the-field) unit summaries + missing regulars."""
    keys = ["game_id", "team"]
    out = lu[keys].drop_duplicates().set_index(keys)
    dbs = lu[lu["grp"].isin(["CB", "S"])]
    for grp, d in (("cb", lu[lu["grp"] == "CB"]), ("s", lu[lu["grp"] == "S"]), ("lb", lu[lu["grp"] == "LB"]), ("db", dbs)):
        out[f"u_{grp}_ypt"] = _wmean(d, "ypt", keys)
        out[f"u_{grp}_cmp"] = _wmean(d, "cmp", keys)
    out["u_db_tdr"] = _wmean(dbs, "tdr", keys)
    out["u_def_mtk"] = _wmean(lu[lu["grp"] != "OL"], "mtk_rate", keys)
    front = lu[lu["grp"].isin(["DL", "LB"])]
    out["u_press_pg"] = (front["press_pg"].fillna(0) * front["w"]).groupby([front[k] for k in keys]).sum()
    cb = lu[lu["grp"] == "CB"].sort_values("w", ascending=False).drop_duplicates(keys).set_index(keys)
    out["u_cb1_ypt"] = cb["ypt"]
    out["u_cb1_share"] = cb["w"]
    # missing regulars: played >= 40% in either of the team's previous two games, absent now
    order = tg[["team", "game_id", "kickoff"]].sort_values(["team", "kickoff"]).copy()
    here = lu[keys + ["pid", "q"]].assign(here=1)
    reg = lu[keys + ["pid", "grp", "w"]]
    cand = []
    for lag in (1, 2):
        order["prev"] = order.groupby("team")["game_id"].shift(lag)
        c = order[["team", "game_id", "prev"]].dropna().merge(
            reg.rename(columns={"game_id": "prev"}), on=["team", "prev"], how="inner")
        c["lag"] = lag
        cand.append(c)
    cand = pd.concat(cand).sort_values("lag").drop_duplicates(keys + ["pid"])
    cand = cand[cand["w"] >= 0.4].merge(here, on=keys + ["pid"], how="left")
    cand["frac"] = np.where(cand["here"].isna(), 1.0, 0.5 * cand["q"].fillna(0))
    miss = cand[cand["frac"] > 0].assign(w=lambda d: d["w"] * d["frac"])
    for grp in ("CB", "S", "LB", "DL", "OL"):
        out[f"u_miss_{grp.lower()}"] = miss[miss["grp"] == grp].groupby(keys)["w"].sum()
    mcols = [c for c in out.columns if c.startswith("u_miss_")]
    out[mcols] = out[mcols].fillna(0.0)
    return out.reset_index()


def pair_features(b: pd.DataFrame, lu: pd.DataFrame) -> pd.DataFrame:
    """Player-vs-specific-defenders history. `b` needs game_id, player_id, opp, kickoff, is_future and
    the residual columns res_*; returns one row per (game_id, player_id)."""
    rcols = [f"res_{v}" for v in RES.values()]
    d = lu[(lu["grp"] != "OL") & (lu["w"] >= 0.3)][["game_id", "team", "pid", "grp", "w"]] \
        .rename(columns={"team": "opp"})
    P = b[["game_id", "player_id", "opp", "kickoff"] + rcols].merge(d, on=["game_id", "opp"], how="inner")
    P = P.sort_values(["kickoff"], kind="stable").reset_index(drop=True)
    key = P["player_id"].astype(str) + "|" + P["pid"].astype(str)
    has = P[rcols[0]].notna().astype(float)
    P["n"] = has.groupby(key, sort=False).cumsum() - has
    for c in rcols:
        P[c + "_cum"] = P[c].fillna(0).groupby(key, sort=False).cumsum() - P[c].fillna(0)
    out = []
    for label, grps in (("db", ["CB", "S"]), ("front", ["DL", "LB"]), ("def", ["CB", "S", "LB", "DL"])):
        q = P[P["grp"].isin(grps)]
        k = [q["game_id"], q["player_id"]]
        W = q["w"].groupby(k).sum().replace(0, np.nan)
        cols = {f"vs{label}_n": (q["w"] * q["n"]).groupby(k).sum() / W}
        for c in rcols:
            cols[f"vs{label}_{c}"] = (q["w"] * q[c + "_cum"] / (q["n"] + 3)).groupby(k).sum() / W
        out.append(pd.DataFrame(cols))
    res = pd.concat(out, axis=1).reset_index()
    res.columns = ["game_id", "player_id"] + list(res.columns[2:])
    keep = ["game_id", "player_id", "vsdb_n", "vsdb_res_rec_yds", "vsdb_res_rec", "vsfront_n",
            "vsfront_res_rush_yds", "vsdef_n", "vsdef_res_pass_yds", "vsdef_res_rec_yds", "vsdef_res_rush_yds"]
    return res[[c for c in keep if c in res.columns]]


def add_defense(b: pd.DataFrame, raw: dict, tg: pd.DataFrame, season: int, log=print) -> pd.DataFrame:
    """Attach opponent-defense unit features (opp_u_*), own offensive-line health (ol_miss) and
    player-vs-defender history (vs*) to the player-game frame `b`."""
    try:
        sn = _defender_form(_snaps(raw, tg))
    except Exception as e:  # keep the pipeline alive if a source is missing
        log(f"  [defense] skipped: {type(e).__name__}: {e}")
        return b
    fut = b.loc[b["is_future"], ["game_id", "team", "opp", "week", "kickoff"]].drop_duplicates(["game_id", "team"])
    fut_teams = pd.concat([fut[["game_id", "team", "week", "kickoff"]],
                           fut[["game_id", "opp", "week", "kickoff"]].rename(columns={"opp": "team"})]).drop_duplicates(["game_id", "team"])
    lu = _lineups(sn, raw, tg, fut_teams, season)
    U = unit_features(lu, tg)
    dcols = [c for c in U.columns if c.startswith("u_") and c != "u_miss_ol"]
    D = U[["game_id", "team"] + dcols].rename(columns={"team": "opp"})
    D.columns = ["game_id", "opp"] + [f"opp_{c}" for c in dcols]
    b = b.merge(D, on=["game_id", "opp"], how="left")
    O = U[["game_id", "team", "u_miss_ol"]].rename(columns={"u_miss_ol": "ol_miss"})
    b = b.merge(O, on=["game_id", "team"], how="left")
    b["ol_miss"] = b["ol_miss"].fillna(0.0)
    pf = pair_features(b, lu)
    b = b.merge(pf, on=["game_id", "player_id"], how="left")
    return b


def opponent_lineup(raw: dict, b: pd.DataFrame, tg: pd.DataFrame, season: int, game_id: str, team: str) -> pd.DataFrame:
    """Readable list of the defenders `team` is expected to field in `game_id` (for the matchup report)."""
    sn = _defender_form(_snaps(raw, tg))
    fut = b.loc[(b["game_id"] == game_id), ["game_id", "team", "week", "kickoff"]].drop_duplicates(["game_id", "team"])
    lu = _lineups(sn, raw, tg, fut, season)
    lu = lu[(lu["game_id"] == game_id) & (lu["team"] == team) & (lu["grp"] != "OL")]
    names = raw["snap_counts"][["pfr_player_id", "player"]].drop_duplicates("pfr_player_id").set_index("pfr_player_id")["player"]
    lu = lu.assign(name=lu["pid"].map(names))
    return lu.sort_values(["grp", "w"], ascending=[True, False])
