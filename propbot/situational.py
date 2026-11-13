"""The remaining parameters from the 250-item checklist that free data supports.

  windows        plain last-3 / last-5 / last-10 averages (the EWMs elsewhere weight recent games smoothly;
                 these are the box-score windows bettors quote), adjusted yards per attempt, yards per
                 reception, touches per snap, 3-game snap share, 5-game usage trend (slope of target / carry share)
  halves         share of a player's yards gained in the first half, and per-half yards
  game script    moneyline win probability (de-vigged), P(blowout), P(one-score game), P(overtime-ish finish)
                 from the spread, expected plays from both teams' pace
  opp defense    third-down conversion rate, red-zone TD rate and sack rate allowed; yards allowed to the
                 opponent's WR1, WR2, TE1, RB1 (roles ranked by pre-game target / carry share)
  availability   games missed this season, weeks on the injury report in the last 17 team games,
                 games since returning from an absence, consecutive road games
  splits         the player's past over/under-performance (vs his own rolling average) in games like this
                 one: home/away, divisional, primetime, cold, bad weather, vs top-10 / bottom-10 defenses,
                 after a loss / win, short rest, as favourite / underdog by a score, with this QB,
                 with a top teammate missing, and over the last 3 meetings with this opponent
Everything is computed from earlier games only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# same-game outcomes created here: never features
OUTCOME_COLS = ["h1_yds", "h2_yds", "res_main", "role_rank"]
SD_MARGIN = 13.5  # NFL final margin vs spread, standard deviation in points


def _shift_roll(b: pd.DataFrame, cols: list[str], n: int) -> pd.DataFrame:
    g = b.groupby("player_id", sort=False)[cols].shift(1)
    r = g.groupby(b["player_id"], sort=False).rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
    r = r.reindex(b.index)
    r.columns = [f"{c}_m{n}" for c in cols]
    return r


def _slope5(y: pd.Series, key: pd.Series) -> pd.Series:
    """Least-squares slope of the last 5 values per key (vectorised rolling sums)."""
    x = y.groupby(key, sort=False).cumcount().astype(float)
    ok = y.notna()
    y0, x0 = y.fillna(0), x.where(ok, 0)
    roll = lambda v: v.groupby(key, sort=False).rolling(5, min_periods=1).sum().reset_index(level=0, drop=True).reindex(y.index)
    n, sx, sy = roll(ok.astype(float)), roll(x0), roll(y0)
    sxy, sxx = roll(x0 * y0), roll(x0 * x0)
    den = n * sxx - sx * sx
    return ((n * sxy - sx * sy) / den.replace(0, np.nan)).where(n >= 3)


def windows(b: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["passing_yards", "attempts", "completions", "passing_tds", "rushing_yards", "carries",
                        "receiving_yards", "receptions", "targets", "rr_yds", "td", "offense_pct"] if c in b.columns]
    b = b.sort_values(["player_id", "kickoff"]).copy()
    out = [_shift_roll(b, cols, n) for n in (3, 5, 10)]
    b = pd.concat([b] + out, axis=1)
    e = lambda c: b.get(f"{c}_e6")
    att = e("attempts").replace(0, np.nan)
    b["aya_e6"] = (e("passing_yards") + 20 * e("passing_tds") - 45 * e("passing_interceptions")) / att
    b["ypr_e6"] = e("receiving_yards") / e("receptions").replace(0, np.nan)
    touches = e("carries").fillna(0) + e("receptions").fillna(0)
    b["touch_per_snap_e6"] = touches / e("offense_snaps").replace(0, np.nan)
    sh = b.groupby("player_id", sort=False)[["target_share", "carry_share"]].shift(1)
    for c in ("target_share", "carry_share"):
        b[f"{c}_trend5"] = _slope5(sh[c], b["player_id"])
    return b


def halves(b: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    if "game_seconds_remaining" not in pbp.columns:
        return b
    p = pbp[(pbp["two_point_attempt"] != 1)]
    first = p["game_seconds_remaining"] > 1800
    parts = []
    for pid, ycol in (("receiver_player_id", "receiving_yards"), ("rusher_player_id", "rushing_yards"),
                      ("passer_player_id", "receiving_yards")):
        d = p[p[pid].notna()]
        parts.append(pd.DataFrame({"game_id": d["game_id"], "player_id": d[pid], "y": d[ycol].fillna(0),
                                   "h1": first[d.index]}))
    d = pd.concat(parts)
    agg = d.groupby(["game_id", "player_id", "h1"])["y"].sum().unstack("h1").fillna(0)
    agg.columns = ["h2_yds" if not c else "h1_yds" for c in agg.columns]
    b = b.merge(agg.reset_index(), on=["game_id", "player_id"], how="left")
    hist = ~b["is_future"]
    b.loc[hist, ["h1_yds", "h2_yds"]] = b.loc[hist, ["h1_yds", "h2_yds"]].fillna(0)
    b = b.sort_values(["player_id", "kickoff"])
    sh = b.groupby("player_id", sort=False)[["h1_yds", "h2_yds"]].shift(1)
    ew = sh.groupby(b["player_id"], sort=False).transform(lambda s: s.ewm(halflife=6, min_periods=1).mean())
    b["h1_yds_e6"], b["h2_yds_e6"] = ew["h1_yds"], ew["h2_yds"]
    b["h1_share_e6"] = b["h1_yds_e6"] / (b["h1_yds_e6"] + b["h2_yds_e6"]).replace(0, np.nan)
    return b


def game_script(b: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    s = sched[["game_id", "home_team", "home_moneyline", "away_moneyline"]].copy()
    imp = lambda ml: np.where(ml > 0, 100 / (ml + 100), -ml / (-ml + 100))
    ph, pa = imp(s["home_moneyline"].astype(float)), imp(s["away_moneyline"].astype(float))
    s["p_home"] = ph / (ph + pa)
    b = b.merge(s[["game_id", "home_team", "p_home"]], on="game_id", how="left")
    b["ml_win_prob"] = np.where(b["team"] == b["home_team"], b["p_home"], 1 - b["p_home"])
    # spread ~ expected margin for this team; final margin ~ N(spread, 13.5)
    m = b["spread"].astype(float)
    cdf = lambda x: norm.cdf((x - m) / SD_MARGIN)
    b["p_blowout"] = 1 - cdf(16.5) + cdf(-16.5)          # either side wins by 17+
    b["p_close"] = cdf(8.5) - cdf(-8.5)                  # one-score game
    b["p_ot_proxy"] = cdf(1.5) - cdf(-1.5)               # regulation finishes within a field goal-ish
    b["p_trail_big"] = cdf(-10.5)                        # this team trails by 11+ at the end (pass-heavy script)
    if "off_plays_e4" in b.columns and "opp_alw_off_plays_e5" in b.columns:
        b["exp_plays"] = (b["off_plays_e4"] + b["opp_alw_off_plays_e5"]) / 2
    return b.drop(columns=["home_team", "p_home"])


def opp_defense(b: pd.DataFrame, pbp: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    p = pbp[(pbp["two_point_attempt"] != 1) & ((pbp["pass"] == 1) | (pbp["rush"] == 1))]
    third = p[p["down"] == 3]
    conv = (third["first_down"].fillna(0).astype(bool) | (third["touchdown"] == 1)).astype(float)
    t3 = conv.groupby([third["game_id"], third["defteam"]]).mean().rename("d3conv")
    rz = p[p["yardline_100"] <= 20]
    rzt = rz["touchdown"].fillna(0).groupby([rz["game_id"], rz["defteam"]]).mean().rename("drz_td")
    db = p[p["qb_dropback"] == 1]
    sk = db["sack"].fillna(0).groupby([db["game_id"], db["defteam"]]).mean().rename("dsack_rate")
    D = pd.concat([t3, rzt, sk], axis=1).reset_index().rename(columns={"defteam": "opp"})
    D = tg[["game_id", "team", "kickoff"]].rename(columns={"team": "opp"}).merge(D, on=["game_id", "opp"], how="left") \
        .sort_values(["opp", "kickoff"])
    for c in ("d3conv", "drz_td", "dsack_rate"):
        D[f"opp_{c}_e5"] = D.groupby("opp")[c].transform(lambda s: s.shift(1).ewm(halflife=5, min_periods=1, ignore_na=True).mean())
    b = b.merge(D[["game_id", "opp", "opp_d3conv_e5", "opp_drz_td_e5", "opp_dsack_rate_e5"]], on=["game_id", "opp"], how="left")

    # role-specific yards allowed: WR1/WR2/TE1/RB1 by pre-game share
    use = np.where(b["position"] == "RB", b.get("carry_share_e6"), b.get("target_share_e6"))
    b["_use"] = pd.Series(use, index=b.index).fillna(0)
    b["role_rank"] = b.groupby(["game_id", "team", "position"])["_use"].rank(ascending=False, method="first")
    roles = {("WR", 1): "wr1", ("WR", 2): "wr2", ("TE", 1): "te1", ("RB", 1): "rb1"}
    b["_role"] = [roles.get((p_, int(r)) if r == r else None) for p_, r in zip(b["position"], b["role_rank"])]
    y = np.where(b["position"] == "RB", b["rr_yds"], b["receiving_yards"])
    R = b.loc[~b["is_future"] & b["_role"].notna()].assign(y=y[(~b["is_future"] & b["_role"].notna()).to_numpy()])
    R = R.pivot_table(index=["game_id", "opp"], columns="_role", values="y", aggfunc="sum")
    R = tg[["game_id", "team", "kickoff", "played"]].rename(columns={"team": "opp"}).merge(
        R.reset_index(), on=["game_id", "opp"], how="left").sort_values(["opp", "kickoff"])
    out = ["game_id", "opp"]
    for r in roles.values():
        if r in R.columns:
            R.loc[R["played"], r] = R.loc[R["played"], r].fillna(0)
            R[f"opp_alw_{r}_e5"] = R.groupby("opp")[r].transform(lambda s: s.shift(1).ewm(halflife=5, min_periods=1, ignore_na=True).mean())
            out.append(f"opp_alw_{r}_e5")
    b = b.merge(R[out], on=["game_id", "opp"], how="left")
    b["opp_alw_my_role_e5"] = np.select([b["_role"] == r for r in roles.values()],
                                        [b.get(f"opp_alw_{r}_e5", np.nan) for r in roles.values()], default=np.nan)
    return b.drop(columns=["_use", "_role"])


def availability(b: pd.DataFrame, raw: dict, tg: pd.DataFrame) -> pd.DataFrame:
    t = tg[["team", "season", "game_id", "kickoff", "is_home"]].sort_values(["team", "kickoff"]).copy()
    t["team_gn"] = t.groupby(["team", "season"]).cumcount()
    # consecutive road games entering this one (this one included if away)
    away = (t["is_home"] < 0.5).astype(int)
    grp = (away != away.groupby(t["team"]).shift()).cumsum()
    t["road_streak"] = away.groupby([t["team"], grp]).cumsum() * away
    b = b.merge(t[["game_id", "team", "team_gn", "road_streak"]], on=["game_id", "team"], how="left")
    b["games_missed_szn"] = (b["team_gn"] - b["games_season"]).clip(lower=0)
    b = b.sort_values(["player_id", "kickoff"])
    gap = b.groupby("player_id", sort=False)["kickoff"].diff().dt.days
    missed_before = (gap > 10).astype(float)  # more than a normal week between his games -> he missed one
    k = (missed_before.groupby(b["player_id"]).cumsum())
    b["games_since_absence"] = b.groupby([b["player_id"], k]).cumcount().astype(float)
    inj = raw["injuries"][["season", "week", "gsis_id"]].dropna().drop_duplicates()
    inj = inj.assign(listed=1.0).rename(columns={"gsis_id": "player_id"})
    inj["season"] = inj["season"].astype(int); inj["week"] = inj["week"].astype(int)
    b = b.merge(inj, on=["season", "week", "player_id"], how="left")
    b["listed"] = b["listed"].fillna(0.0)
    b["inj_hist17"] = b.groupby("player_id", sort=False)["listed"].transform(lambda s: s.shift(1).rolling(17, min_periods=1).sum())
    return b.drop(columns=["team_gn", "listed"])


def splits(b: pd.DataFrame) -> pd.DataFrame:
    """Prior shrunk mean residual under the current value of each condition."""
    main = np.where(b["position"] == "QB", b.get("res_pass_yds"), b.get("res_rr_yds"))
    b["res_main"] = main
    temp = b["wx_temp"] if "wx_temp" in b.columns else b.get("temp")
    tier_src = b.get("opp_pts_allowed_e5")
    tier = pd.Series(np.nan, index=b.index)
    if tier_src is not None:
        r = tier_src.groupby([b["season"], b["week"]]).rank(pct=True)
        tier = pd.Series(np.select([r <= 10 / 32, r >= 22 / 32], [1, -1], 0), index=b.index).where(r.notna())
    T = b[["game_id", "team", "kickoff"]].drop_duplicates(["game_id", "team"]).sort_values(["team", "kickoff"])
    last = b.groupby(["game_id", "team"])["rec_streak"].first() if "rec_streak" in b.columns else None
    cond = {
        "home": (b["is_home"] >= 0.5).astype(int),
        "div": b["div_game"].fillna(0).astype(int),
        "prime": b.get("primetime", pd.Series(0, index=b.index)).fillna(0).astype(int),
        "cold": (temp < 40).astype(int).where(temp.notna(), -1) if temp is not None else pd.Series(-1, index=b.index),
        "badwx": b.get("wx_bad", pd.Series(np.nan, index=b.index)).fillna(-1).astype(int),
        "deftier": tier.fillna(9).astype(int),
        "afterloss": (b["rec_streak"] < 0).astype(int) if "rec_streak" in b.columns else pd.Series(0, index=b.index),
        "shortrest": (b["rest"] <= 6).astype(int),
        "script": pd.cut(b["spread"].fillna(0), [-99, -6.5, 6.5, 99], labels=[0, 1, 2]).astype(int),
        "qb": b["qb_id"].fillna("?").astype(str) if "qb_id" in b.columns else pd.Series("?", index=b.index),
        "mate_out": (b["vac_tgt"] + b["vac_carry"] > 0.15).astype(int),
    }
    d = b[["player_id", "kickoff", "res_main"]].copy()
    order = d.sort_values("kickoff", kind="stable").index
    for name, c in cond.items():
        key = (d["player_id"].astype(str) + "|" + c.astype(str)).loc[order]
        v = d.loc[order, "res_main"]
        has = v.notna().astype(float)
        n = has.groupby(key, sort=False).cumsum() - has
        s = v.fillna(0).groupby(key, sort=False).cumsum() - v.fillna(0)
        b.loc[order, f"sp_{name}_res"] = (s / (n + 3)).to_numpy()
        b.loc[order, f"sp_{name}_n"] = n.to_numpy()
    # last three meetings with this opponent
    key = (d["player_id"].astype(str) + "|" + b["opp"].astype(str)).loc[order]
    v = d.loc[order, "res_main"]
    b.loc[order, "vsopp_last3_res"] = v.groupby(key, sort=False).transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()).to_numpy()
    return b


def add(b: pd.DataFrame, raw: dict, tg: pd.DataFrame, log=print) -> pd.DataFrame:
    steps = (("windows", lambda x: windows(x)), ("halves", lambda x: halves(x, raw["pbp"])),
             ("game_script", lambda x: game_script(x, raw["schedules"])), ("opp_defense", lambda x: opp_defense(x, raw["pbp"], tg)),
             ("availability", lambda x: availability(x, raw, tg)), ("splits", lambda x: splits(x)))
    for name, fn in steps:
        try:
            b = fn(b)
        except Exception as e:  # one missing source must not take down the whole frame
            log(f"  [situational] {name} skipped: {type(e).__name__}: {e}")
    return b
