"""nflverse data access with a per-season parquet cache (no API key needed)."""
from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import polars as pl

from .config import DATA, FIRST_SEASON, RAW, TEAM_FIX

warnings.filterwarnings("ignore")

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "pass", "rush",
    "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "sack", "interception", "yardline_100", "air_yards",
    "complete_pass", "receiving_yards", "rushing_yards", "yards_gained", "touchdown", "epa", "success",
    "pass_oe", "xpass", "down", "two_point_attempt", "receiver_player_id", "rusher_player_id",
    "passer_player_id", "td_player_id", "td_team", "wp", "half_seconds_remaining",
    "shotgun", "no_huddle", "cp", "xyac_mean_yardage", "yards_after_catch", "qb_hit", "first_down",
    "score_differential", "game_seconds_remaining", "ydstogo", "tackled_for_loss", "pass_length",
]
# bump when the cached columns of a dataset change, so stale parquet files are not reused
CACHE_VERSION = {"pbp": "v2"}
# first season each optional dataset exists for
FIRST_AVAILABLE = {"ngs_receiving": 2016, "ngs_rushing": 2016, "ngs_passing": 2016,
                   "pfr_rec": 2018, "pfr_rush": 2018, "pfr_pass": 2018, "pfr_def": 2018}
FFOPP_COLS = ["season", "week", "game_id", "player_id", "receptions_exp", "rec_yards_gained_exp", "rec_touchdown_exp",
              "rush_yards_gained_exp", "rush_touchdown_exp", "pass_yards_gained_exp", "pass_touchdown_exp",
              "pass_completions_exp", "pass_interception_exp", "total_first_down_exp", "total_fantasy_points_exp"]


def _raw_download(fname: str, dest):
    """dynastyprocess files straight from raw.githubusercontent.com (nflreadpy goes via github.com/.../raw,
    which some networks refuse)."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"https://raw.githubusercontent.com/dynastyprocess/data/master/files/{fname}", timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)


def current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def _fetch(name: str, season: int | None) -> pl.DataFrame:
    import nflreadpy as nfl
    if name == "player_stats":
        return nfl.load_player_stats([season])
    if name == "snap_counts":
        return nfl.load_snap_counts([season])
    if name == "injuries":
        return nfl.load_injuries([season])
    if name == "rosters_weekly":
        return nfl.load_rosters_weekly([season])
    if name == "pbp":
        df = nfl.load_pbp([season])
        return df.select([c for c in PBP_COLS if c in df.columns])
    if name == "ff_opportunity":
        df = nfl.load_ff_opportunity([season])
        return df.select([c for c in FFOPP_COLS if c in df.columns]).with_columns(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    if name.startswith("ngs_"):
        return nfl.load_nextgen_stats([season], name[4:]).filter(pl.col("week") > 0)
    if name.startswith("pfr_"):
        return nfl.load_pfr_advstats([season], name[4:], "week")
    if name == "depth_charts":
        # two feeds: weekly team depth charts through 2024, daily ESPN snapshots since. Reduce both to
        # (player, depth_rank) where 1 = starter; the snapshots keep their timestamp instead of a week.
        df = nfl.load_depth_charts([season])
        skill = ["QB", "RB", "FB", "WR", "TE"]
        if "dt" in df.columns:
            df = df.filter(pl.col("pos_abb").is_in(skill) & pl.col("gsis_id").is_not_null())
            rank = pl.when(pl.col("pos_abb") == "WR").then((pl.col("pos_rank") - 1) // 3 + 1).otherwise(pl.col("pos_rank"))
            return df.select(pl.lit(None, dtype=pl.Int32).alias("season"), pl.lit(None, dtype=pl.Int32).alias("week"),
                             pl.col("team"), pl.col("gsis_id").alias("player_id"), rank.cast(pl.Float64).alias("depth_rank"),
                             pl.col("dt")).group_by(["season", "week", "team", "player_id", "dt"]).agg(pl.col("depth_rank").min())
        df = df.filter((pl.col("formation") == "Offense") & pl.col("position").is_in(skill) & pl.col("gsis_id").is_not_null()
                       & pl.col("week").is_not_null())
        return df.group_by(["season", "week", "club_code", "gsis_id"]).agg(pl.col("depth_team").cast(pl.Float64).min().alias("depth_rank")) \
            .select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), pl.col("club_code").alias("team"),
                    pl.col("gsis_id").alias("player_id"), pl.lit(None, dtype=pl.Utf8).alias("dt"), pl.col("depth_rank"))
    if name == "ff_rankings":
        # FantasyPros expert consensus ranks (dynastyprocess archive, 2020+): weekly and rest-of-season
        # positional ranks with their scrape date. A local copy in data/external/ is used when the download
        # host is not reachable.
        local = DATA / "external" / "db_fpecr.parquet"
        try:
            df = nfl.load_ff_rankings("all")
        except Exception:
            if not local.exists():
                _raw_download("db_fpecr.parquet", local)
            df = pl.read_parquet(local)
        keep = [f"{k}-{p}" for k in ("weekly", "redraft") for p in ("qb", "rb", "wr", "te")]
        return df.filter(pl.col("page_type").is_in(keep) & pl.col("ecr_type").is_in(["wp", "rp"])) \
            .select(pl.col("id").cast(pl.Utf8).alias("fp_id"), "mergename", "pos", "page_type",
                    pl.col("ecr").cast(pl.Float64), pl.col("sd").cast(pl.Float64), pl.col("best").cast(pl.Float64),
                    pl.col("worst").cast(pl.Float64), pl.col("scrape_date").cast(pl.Utf8))
    if name == "ff_playerids":
        local = DATA / "external" / "db_playerids.csv"
        try:
            df = nfl.load_ff_playerids()
        except Exception:
            if not local.exists():
                _raw_download("db_playerids.csv", local)
            df = pl.read_csv(local, infer_schema_length=0)
        return df.select(pl.col("fantasypros_id").cast(pl.Utf8).alias("fp_id"), pl.col("gsis_id").cast(pl.Utf8),
                         pl.col("merge_name").cast(pl.Utf8), pl.col("position").cast(pl.Utf8)).filter(pl.col("gsis_id").is_not_null())
    if name == "schedules":
        return nfl.load_schedules()
    if name == "players":
        return nfl.load_players()
    raise ValueError(name)


def load(name: str, seasons: list[int] | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load a dataset; past seasons are cached forever, `refresh` re-pulls the current season
    and the season-less tables (schedules, players)."""
    cur = current_season()
    frames = []
    tag = name + ("_" + CACHE_VERSION[name] if name in CACHE_VERSION else "")
    for s in (seasons or [None]):
        if s is not None and s < FIRST_AVAILABLE.get(name, 0):
            continue
        path = RAW / (f"{tag}_{s}.parquet" if s else f"{tag}.parquet")
        stale = refresh and (s is None or s >= cur)
        if path.exists() and not stale:
            frames.append(pl.read_parquet(path))
            continue
        try:
            df = _fetch(name, s)
        except Exception as e:  # e.g. a dataset not published yet for this season
            if path.exists():
                frames.append(pl.read_parquet(path))
            else:
                print(f"  [data] {name} {s}: unavailable ({type(e).__name__})")
            continue
        df.write_parquet(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pl.concat(frames, how="diagonal_relaxed").to_pandas()
    for c in ("team", "opponent_team", "opponent", "posteam", "defteam", "home_team", "away_team", "td_team"):
        if c in out.columns:
            out[c] = out[c].replace(TEAM_FIX)
    return out


def load_all(refresh: bool = False, first: int = FIRST_SEASON) -> dict[str, pd.DataFrame]:
    seasons = list(range(first, current_season() + 1))
    raw = {
        "schedules": load("schedules", refresh=refresh),
        "players": load("players", refresh=refresh),
        "player_stats": load("player_stats", seasons, refresh),
        "snap_counts": load("snap_counts", seasons, refresh),
        "injuries": load("injuries", seasons, refresh),
        "pbp": load("pbp", seasons, refresh),
        "rosters_weekly": load("rosters_weekly", [current_season()], refresh),
    }
    for extra in ("ff_opportunity", "ngs_receiving", "ngs_rushing", "ngs_passing",
                  "pfr_rec", "pfr_rush", "pfr_pass", "pfr_def", "depth_charts"):
        raw[extra] = load(extra, seasons, refresh)
    raw["ff_rankings"] = load("ff_rankings", refresh=refresh)
    raw["ff_playerids"] = load("ff_playerids", refresh=refresh)
    return raw
