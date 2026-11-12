"""Game-day weather from Open-Meteo (free, no key).

* History: ERA5 reanalysis via the archive API, so the model can *learn* what rain, gusts and snow do.
* Upcoming games: the forecast API (up to 16 days out), so the same features exist at prediction time.

Each game gets the hours from kickoff to kickoff + 3h at the stadium:
temperature (F), sustained wind and gusts (mph), precipitation and snowfall (inches), humidity.
Domes / closed roofs get neutral values. Also exposes venue coordinates for travel and altitude.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import requests

from .config import RAW

# current home stadium per team: (lat, lon, elevation_m)
HOME = {
    "ARI": (33.5276, -112.2626, 331), "ATL": (33.7554, -84.4009, 300), "BAL": (39.2780, -76.6227, 10),
    "BUF": (42.7738, -78.7870, 200), "CAR": (35.2258, -80.8528, 230), "CHI": (41.8623, -87.6167, 180),
    "CIN": (39.0955, -84.5161, 150), "CLE": (41.5061, -81.6995, 180), "DAL": (32.7473, -97.0945, 170),
    "DEN": (39.7439, -105.0201, 1609), "DET": (42.3400, -83.0456, 180), "GB": (44.5013, -88.0622, 200),
    "HOU": (29.6847, -95.4107, 15), "IND": (39.7601, -86.1639, 220), "JAX": (30.3239, -81.6373, 5),
    "KC": (39.0489, -94.4839, 270), "LV": (36.0909, -115.1833, 610), "LAC": (33.9535, -118.3392, 30),
    "LA": (33.9535, -118.3392, 30), "MIA": (25.9580, -80.2389, 2), "MIN": (44.9737, -93.2577, 250),
    "NE": (42.0909, -71.2643, 80), "NO": (29.9511, -90.0812, 1), "NYG": (40.8135, -74.0745, 3),
    "NYJ": (40.8135, -74.0745, 3), "PHI": (39.9008, -75.1675, 5), "PIT": (40.4468, -80.0158, 220),
    "SEA": (47.5952, -122.3316, 5), "SF": (37.4030, -121.9700, 5), "TB": (27.9759, -82.5033, 10),
    "TEN": (36.1665, -86.7713, 130), "WAS": (38.9077, -76.8645, 60),
}
# neutral sites and former stadiums, matched by substring of the schedule's stadium name
NAMED = [
    ("wembley", (51.5560, -0.2795, 40)), ("twickenham", (51.4560, -0.3415, 10)),
    ("tottenham", (51.6043, -0.0664, 30)), ("azteca", (19.3029, -99.1505, 2240)),
    ("banorte", (19.3029, -99.1505, 2240)), ("allianz", (48.2188, 11.6247, 510)),
    ("frankfurt", (50.0686, 8.6455, 110)), ("deutsche bank", (50.0686, 8.6455, 110)),
    ("corinthians", (-23.5453, -46.4742, 760)), ("neo qu", (-23.5453, -46.4742, 760)),
    ("bernab", (40.4531, -3.6883, 650)), ("croke", (53.3607, -6.2512, 20)), ("maracan", (-22.9122, -43.2302, 10)),
    ("melbourne", (-37.8200, 144.9834, 10)), ("olympiastadion", (52.5147, 13.2395, 50)),
    ("coliseum", None), ("stubhub", (33.8644, -118.2611, 15)), ("dignity", (33.8644, -118.2611, 15)),
    ("qualcomm", (32.7831, -117.1196, 20)), ("sdccu", (32.7831, -117.1196, 20)),
    ("edward jones", (38.6328, -90.1885, 140)), ("georgia dome", (33.7577, -84.4008, 300)),
    ("hubert h", (44.9740, -93.2581, 250)), ("tcf bank", (44.9765, -93.2245, 250)),
    ("rogers centre", (43.6414, -79.3894, 80)), ("stade de france", (48.9245, 2.3602, 40)),
    ("bayern", (48.2188, 11.6247, 510)),
]
# another team's stadium used as a neutral site (Super Bowls, relocated games)
NAME_TO_TEAM = [
    ("levi", "SF"), ("nrg", "HOU"), ("reliant", "HOU"), ("u.s. bank", "MIN"), ("mercedes-benz stadium", "ATL"),
    ("superdome", "NO"), ("hard rock", "MIA"), ("sun life", "MIA"), ("state farm", "ARI"), ("phoenix", "ARI"),
    ("raymond james", "TB"), ("sofi", "LA"), ("ford field", "DET"), ("allegiant", "LV"), ("acrisure", "PIT"),
    ("heinz", "PIT"), ("firstenergy", "CLE"), ("huntington bank field", "CLE"), ("cleveland browns", "CLE"),
    ("metlife", "NYJ"), ("lucas oil", "IND"), ("tiaa", "JAX"), ("everbank", "JAX"), ("arrowhead", "KC"),
    ("lambeau", "GB"), ("soldier", "CHI"), ("gillette", "NE"), ("at&t", "DAL"), ("lincoln financial", "PHI"),
]
OAKLAND = (37.7516, -122.2005, 10)
LA_COLISEUM = (34.0141, -118.2879, 60)
WX_COLS = ["wx_temp", "wx_wind", "wx_gust", "wx_precip", "wx_snow", "wx_humidity"]
HOURLY = "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation,snowfall,relative_humidity_2m"
UNITS = dict(temperature_unit="fahrenheit", wind_speed_unit="mph", precipitation_unit="inch", timezone="America/New_York")


def venue(home_team: str, season: int, stadium: str | None, location: str | None) -> tuple[float, float, float]:
    name = str(stadium or "").lower()
    for key, v in NAMED:
        if key in name:
            if key == "coliseum":
                if "oakland" in name or "ringcentral" in name or (home_team == "LV" and season < 2020):
                    return OAKLAND
                return LA_COLISEUM
            return v
    for key, t in NAME_TO_TEAM:
        if key in name:
            return HOME[t]
    if home_team == "LV" and season < 2020:
        return OAKLAND
    if home_team == "LAC" and season < 2017:
        return (32.7831, -117.1196, 20)
    if home_team == "LA" and season < 2016:
        return (38.6328, -90.1885, 140)
    return HOME.get(home_team, (np.nan, np.nan, np.nan))


def venues(sched: pd.DataFrame) -> pd.DataFrame:
    """game_id -> lat, lon, elevation of the venue."""
    v = [venue(h, s, st, loc) for h, s, st, loc in
         zip(sched["home_team"], sched["season"], sched.get("stadium"), sched.get("location"))]
    return pd.DataFrame({"game_id": sched["game_id"].to_numpy(), "v_lat": [x[0] for x in v],
                         "v_lon": [x[1] for x in v], "v_elev": [x[2] for x in v]})


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- fetching
class QuotaExceeded(RuntimeError):
    pass


def _get(url: str, params: dict, tries: int = 4) -> dict | None:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 + 2 * i)
            continue
        if r.status_code == 429:
            if "Daily" in r.text or "Hourly" in r.text:  # quota gone: retrying only wastes time
                raise QuotaExceeded(r.text[:120])
            time.sleep(5 * (i + 1))
            continue
        if r.ok:
            return r.json()
        return None
    return None


def _hourly(js: dict | None) -> pd.DataFrame:
    if not js or "hourly" not in js:
        return pd.DataFrame()
    h = pd.DataFrame(js["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h


def _summarise(h: pd.DataFrame, kickoff: pd.Timestamp) -> dict:
    w = h[(h["time"] >= kickoff.floor("h")) & (h["time"] <= kickoff.floor("h") + pd.Timedelta(hours=3))]
    if w.empty:
        return {}
    return dict(wx_temp=w["temperature_2m"].mean(), wx_wind=w["wind_speed_10m"].max(),
                wx_gust=w["wind_gusts_10m"].max(), wx_precip=w["precipitation"].sum(),
                wx_snow=w["snowfall"].sum(), wx_humidity=w["relative_humidity_2m"].mean())


def _games(sched: pd.DataFrame) -> pd.DataFrame:
    g = sched.merge(venues(sched), on="game_id")
    g["kickoff"] = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].fillna("13:00").astype(str), errors="coerce")
    g["indoor"] = g["roof"].isin(["dome", "closed"])
    return g.dropna(subset=["kickoff", "v_lat"])


def history(sched: pd.DataFrame, refresh: bool = False, log=print) -> pd.DataFrame:
    """Observed weather for played outdoor games (cached; only new games are fetched)."""
    path = RAW / "weather_hist.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["game_id"] + WX_COLS)
    g = _games(sched)
    today = pd.Timestamp.now().normalize()
    todo = g[g["home_score"].notna() & ~g["indoor"] & ~g["game_id"].isin(have["game_id"])
             & (g["kickoff"] < today - pd.Timedelta(days=6))]  # ERA5 lags ~5 days
    if todo.empty:
        return have
    log(f"  [weather] fetching archive weather for {len(todo)} games")
    todo = todo.assign(month=todo["kickoff"].dt.to_period("M"))
    jobs = list(todo.groupby(["v_lat", "v_lon", "month"]))

    def one(job):
        (lat, lon, _), grp = job
        start, end = grp["kickoff"].min().date(), (grp["kickoff"].max() + pd.Timedelta(days=1)).date()
        h = _hourly(_get("https://archive-api.open-meteo.com/v1/archive", dict(
            latitude=lat, longitude=lon, start_date=str(start), end_date=str(end), hourly=HOURLY, **UNITS)))
        if h.empty:
            return []
        return [dict(game_id=gid, **s) for gid, k in zip(grp["game_id"], grp["kickoff"]) if (s := _summarise(h, k))]

    from concurrent.futures import ThreadPoolExecutor
    rows = []
    with ThreadPoolExecutor(4) as ex:  # one request per stadium-month: long ranges time out
        for i, r in enumerate(ex.map(one, jobs), 1):
            rows += r
            if i % 200 == 0 or i == len(jobs):
                have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("game_id", keep="last")
                have.to_parquet(path)
                rows = []
                log(f"  [weather] {i}/{len(jobs)} requests, {len(have)} games cached")
    return have


def forecast(sched: pd.DataFrame, days: int = 16, log=print) -> pd.DataFrame:
    """Forecast for unplayed outdoor games kicking off in the next `days` days (cached for 2 hours)."""
    path = RAW / "weather_fcst.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < 7200:
        return pd.read_parquet(path)
    g = _games(sched)
    now = pd.Timestamp.now()
    up = g[g["home_score"].isna() & ~g["indoor"] & (g["kickoff"] > now - pd.Timedelta(hours=4))
           & (g["kickoff"] < now + pd.Timedelta(days=days))]
    rows = []
    for (lat, lon), grp in up.groupby(["v_lat", "v_lon"]):
        try:
            h = _hourly(_get("https://api.open-meteo.com/v1/forecast", dict(
                latitude=lat, longitude=lon, hourly=HOURLY, forecast_days=16, **UNITS)))
        except QuotaExceeded as e:
            log(f"  [weather] forecast quota exhausted ({e}); upcoming games get no weather this run")
            break
        for gid, k in zip(grp["game_id"], grp["kickoff"]):
            s = _summarise(h, k) if len(h) else {}
            if s:
                rows.append(dict(game_id=gid, **s))
    out = pd.DataFrame(rows, columns=["game_id"] + WX_COLS)
    if len(up):
        log(f"  [weather] forecast for {len(out)}/{len(up)} upcoming outdoor games")
    out.to_parquet(path)
    return out


def game_weather(sched: pd.DataFrame, refresh: bool = False, online: bool = True, backfill: bool = False,
                 log=print) -> pd.DataFrame:
    """One row per game with WX_COLS (+ venue). Indoor games get neutral values; history falls back to
    the schedule's own temp/wind when the archive has nothing."""
    g = _games(sched)[["game_id", "season", "kickoff", "indoor", "temp", "wind", "v_lat", "v_lon", "v_elev"]]
    p = RAW / "weather_hist.parquet"
    parts = [pd.read_parquet(p)] if p.exists() else []
    if online:
        try:
            if backfill:  # slow (one request per stadium-month); run `propbot weather-backfill`
                parts.append(history(sched, refresh, log))
            parts.append(forecast(sched, log=log))
        except Exception as e:  # weather is optional: never take training/scanning down
            log(f"  [weather] unavailable: {type(e).__name__}: {e}")
    wx = pd.concat([p for p in parts if len(p)], ignore_index=True).drop_duplicates("game_id", keep="last") \
        if any(len(p) for p in parts) else pd.DataFrame(columns=["game_id"] + WX_COLS)
    out = g.merge(wx, on="game_id", how="left")
    out["wx_temp"] = out["wx_temp"].fillna(out["temp"])
    out["wx_wind"] = out["wx_wind"].fillna(out["wind"])
    ind = out["indoor"]
    out.loc[ind, "wx_temp"] = 70.0
    out.loc[ind, ["wx_wind", "wx_gust", "wx_precip", "wx_snow"]] = 0.0
    out.loc[ind, "wx_humidity"] = 45.0
    out["wx_cold"] = (out["wx_temp"] < 32).astype(float).where(out["wx_temp"].notna())
    out["wx_bad"] = ((out["wx_wind"] >= 15) | (out["wx_gust"] >= 25) | (out["wx_precip"] >= 0.1) |
                     (out["wx_snow"] > 0)).astype(float).where(out["wx_wind"].notna())
    return out[["game_id", "v_lat", "v_lon", "v_elev"] + WX_COLS + ["wx_cold", "wx_bad"]]
