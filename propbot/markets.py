"""Turn Kalshi NFL prop markets into rows the model can price, and map them to nflverse players."""
from __future__ import annotations

import difflib
import re
import unicodedata

import pandas as pd

from .config import SERIES_TO_STAT, TEAM_FIX

KALSHI_TEAM = dict(TEAM_FIX)  # LAR->LA, JAC->JAX, ...
NFL_TEAMS = {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX",
             "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"}
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’`-]", "", s)
    return re.sub(r"\s+", " ", _SUFFIX.sub("", s)).strip()


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def split_teams(code: str) -> tuple[str, str] | None:
    """'BUFNYJ' -> ('BUF','NYJ') (away, home), using Kalshi's own abbreviations."""
    for i in (3, 2):
        a, h = code[:i], code[i:]
        if KALSHI_TEAM.get(a, a) in NFL_TEAMS and KALSHI_TEAM.get(h, h) in NFL_TEAMS:
            return KALSHI_TEAM.get(a, a), KALSHI_TEAM.get(h, h)
    return None


def parse(markets: list[dict]) -> pd.DataFrame:
    rows = []
    for m in markets:
        parts = m["ticker"].split("-")
        stat = SERIES_TO_STAT.get(parts[0])
        if stat is None or len(parts) < 4 or m.get("strike_type") != "greater" or m.get("floor_strike") is None:
            continue
        teams = split_teams(parts[1][7:])
        if not teams:
            continue
        away, home = teams
        pc = parts[2]
        team = next((KALSHI_TEAM.get(t, t) for t in (pc[:3], pc[:2]) if KALSHI_TEAM.get(t, t) in (away, home)), None)
        yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
        nb, na = _f(m.get("no_bid_dollars")), _f(m.get("no_ask_dollars"))
        rows.append(dict(
            ticker=m["ticker"], event=m["event_ticker"], series=parts[0], stat=stat, away=away, home=home, team=team,
            player=str(m.get("title", "")).split(":")[0].strip(), strike=float(m["floor_strike"]),
            yes_bid=yb, yes_ask=ya, no_bid=nb, no_ask=na,
            yes_ask_size=_f(m.get("yes_ask_size_fp")), no_ask_size=_f(m.get("yes_bid_size_fp")),
            volume=_f(m.get("volume_fp")), open_interest=_f(m.get("open_interest_fp")),
            status=m.get("status"), result=m.get("result"), settle_value=_f(m.get("settlement_value_dollars")),
            open_time=m.get("open_time"), close_time=m.get("close_time")))
    return pd.DataFrame(rows)


def attach_players(mk: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Add game_id + player_id by matching (away, home) to a game and the name within the team.
    `frame` is the feature frame restricted to the candidate games (future rows for a live scan)."""
    if mk.empty:
        return mk.assign(game_id=None, player_id=None)
    games = frame[["game_id", "team", "opp", "is_home"]].drop_duplicates()
    home = games[games["is_home"] >= 0.5].rename(columns={"team": "home", "opp": "away"})[["game_id", "home", "away"]]
    away = games[games["is_home"] <= 0.5].rename(columns={"team": "away", "opp": "home"})[["game_id", "home", "away"]]
    gmap = pd.concat([home, away]).drop_duplicates(["home", "away"])
    out = mk.merge(gmap, on=["home", "away"], how="left")
    roster = frame[["game_id", "team", "player_id", "name"]].dropna(subset=["name"]).copy()
    roster["nn"] = roster["name"].map(norm_name)
    lookup = {(g, t): d for (g, t), d in roster.groupby(["game_id", "team"])}
    ids, cache = [], {}
    for g, t, name in zip(out["game_id"], out["team"], out["player"]):
        key = (g, t, name)
        if key not in cache:
            pid = None
            d = lookup.get((g, t))
            if d is not None:
                nn = norm_name(name)
                hit = d[d["nn"] == nn]
                if hit.empty:  # same surname + first initial, then fuzzy
                    last, first = nn.split(" ")[-1], nn[:1]
                    hit = d[d["nn"].str.split(" ").str[-1].eq(last) & d["nn"].str[:1].eq(first)]
                if hit.empty:
                    close = difflib.get_close_matches(nn, list(d["nn"]), n=1, cutoff=0.86)
                    hit = d[d["nn"].isin(close)]
                if len(hit) == 1:
                    pid = hit["player_id"].iloc[0]
            cache[key] = pid
        ids.append(cache[key])
    out["player_id"] = ids
    return out
