"""nflverse data access with a per-season parquet cache (no API key needed)."""
from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import polars as pl

from .config import FIRST_SEASON, RAW, TEAM_FIX

warnings.filterwarnings("ignore")

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "pass", "rush",
    "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "sack", "interception", "yardline_100", "air_yards",
    "complete_pass", "receiving_yards", "rushing_yards", "yards_gained", "touchdown", "epa", "success",
    "pass_oe", "xpass", "down", "two_point_attempt", "receiver_player_id", "rusher_player_id",
    "passer_player_id", "td_player_id", "td_team", "wp", "half_seconds_remaining",
]


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
    for s in (seasons or [None]):
        path = RAW / (f"{name}_{s}.parquet" if s else f"{name}.parquet")
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
    return raw
