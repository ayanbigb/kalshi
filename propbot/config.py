"""Central configuration: paths, stat definitions, Kalshi series mapping, risk defaults."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for _p in (RAW, MODELS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

FIRST_SEASON = 2015          # first season pulled (used only to warm up rolling features)
TRAIN_START = 2016           # first season whose rows are used for training
STACK_START = 2018           # first season with out-of-fold stage-1 projections
BACKTEST_SEASONS = [2023, 2024, 2025, 2026]

SKILL_POS = ["QB", "RB", "FB", "WR", "TE"]
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}

_YDS = [9.5, 14.5, 19.5, 24.5, 29.5, 34.5, 39.5, 44.5, 49.5, 54.5, 59.5, 64.5, 69.5, 79.5, 89.5,
        99.5, 109.5, 119.5, 129.5, 139.5, 149.5, 169.5, 189.5]

# stat key -> definition.  `grid` = thresholds used to train the ladder model
# (the threshold is a numeric feature, so any strike Kalshi lists can be priced).
STATS: dict[str, dict] = {
    "pass_yds": dict(col="passing_yards", series="KXNFLPASSYDS", pos=["QB"], starter_qb=True, obj="tweedie",
                     grid=[99.5, 124.5, 149.5, 174.5, 199.5, 224.5, 249.5, 274.5, 299.5, 324.5, 349.5, 374.5, 399.5, 424.5]),
    "pass_tds": dict(col="passing_tds", series="KXNFLPASSTDS", pos=["QB"], starter_qb=True, obj="poisson",
                     grid=[0.5, 1.5, 2.5, 3.5, 4.5]),
    "pass_att": dict(col="attempts", series="KXNFLPASSATT", pos=["QB"], starter_qb=True, obj="poisson",
                     grid=[17.5, 20.5, 23.5, 26.5, 28.5, 30.5, 31.5, 33.5, 35.5, 36.5, 38.5, 41.5, 44.5, 48.5]),
    "pass_comp": dict(col="completions", series="KXNFLPASSCOMP", pos=["QB"], starter_qb=True, obj="poisson",
                      grid=[9.5, 11.5, 13.5, 14.5, 16.5, 18.5, 19.5, 21.5, 23.5, 24.5, 26.5, 28.5, 31.5, 34.5]),
    "pass_int": dict(col="passing_interceptions", series="KXNFLPASSINT", pos=["QB"], starter_qb=True, obj="poisson",
                     grid=[0.5, 1.5, 2.5]),
    "rush_yds": dict(col="rushing_yards", series="KXNFLRSHYDS", pos=["RB", "FB", "QB", "WR"], obj="tweedie",
                     grid=[4.5] + _YDS),
    "rush_att": dict(col="carries", series="KXNFLRSHATT", pos=["RB", "FB", "QB", "WR"], obj="poisson",
                     grid=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 16.5, 18.5, 20.5, 23.5, 26.5]),
    "rec": dict(col="receptions", series="KXNFLREC", pos=["WR", "TE", "RB", "FB"], obj="poisson",
                grid=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]),
    "rec_yds": dict(col="receiving_yards", series="KXNFLRECYDS", pos=["WR", "TE", "RB", "FB"], obj="tweedie",
                    grid=_YDS),
    "rr_yds": dict(col="rr_yds", series="KXNFLRRYDS", pos=["RB", "FB", "WR", "TE"], obj="tweedie",
                   grid=[19.5, 29.5, 39.5, 49.5, 59.5, 64.5, 69.5, 79.5, 84.5, 89.5, 99.5, 109.5, 119.5, 129.5, 149.5, 169.5, 189.5]),
    "td": dict(col="td", series="KXNFLTD", pos=["RB", "FB", "WR", "TE", "QB"], obj="poisson",
               grid=[0.5, 1.5, 2.5]),
}
SERIES_TO_STAT = {v["series"]: k for k, v in STATS.items()}

DEFAULTS = {
    "kalshi": {
        "env": "prod",                # prod | demo  (market data is always public prod)
        "series": [v["series"] for v in STATS.values()],
    },
    "model": {
        "market_weight": 0.7,         # shrink model prob toward the market mid (0 = trust model fully)
        "book_weight": 0.5,           # if sportsbook props are available, weight on de-vigged book prob
        "learned_blend": False,       # use models/blend.json instead of market_weight. Off: over 2025-26 the
                                      # fixed weight beat the learned one walk-forward (see RESEARCH.md)
        "blend_ridge": 3000.0,
        "use_weather": False,
        # Features left out of the PRICE model: the FTN charting / participation set (tied on the market board,
        # RESEARCH.md) and the matchup / situational set: on the market board it
        # made the model slightly worse (0.5054 -> 0.5064 log-loss, blend gain +2.37 -> +2.27; RESEARCH.md).
        # They are still built and used by `matchup`, the combo engine and PARAMETERS.md. None = use everything.
        "drop_feats": r"^(sp_|vs|opp_u_|ol_miss|rec_(gp|wpct|pd_pg|pd_l3|streak)|opp_rec_|wpct_diff|ref_tot|travel_km|tz_shift|body_clock|altitude|primetime|early_window|thursday|short_week|rest_edge|hc_new|aya_|ypr_e6|touch_per|h1_|h2_|ml_win|p_blowout|p_close|p_ot|p_trail|exp_plays|opp_d3conv|opp_drz_td|opp_dsack_rate|opp_alw_(wr|te|rb|my)|road_streak|games_missed|games_since|inj_hist|role_rank)|_m(3|5|10)$|_trend5$|^ftn_|^opp_ftn_|^route_part_prev$|^tprr_car$|^yprr_car$|^ypt_(man|zone)_car$|^man_minus_zone$|^opp_man_rate_prev$|^man_zone_matchup$",         # wx_* features in the model (needs forecasts at scan time; see weather.py)        # shrinkage of the learned blend toward market_weight
    },
    "risk": {
        "bankroll": 500.0,            # paper bankroll; live mode reads the real balance
        "kelly_fraction": 0.20,
        "min_edge": 0.025,            # min expected profit per contract after fees, in dollars
        "min_roi": 0.05,              # min edge / cost
        "max_disagreement": 0.20,     # |model - market| above this is treated as missing news, not edge
        "min_price": 0.08, "max_price": 0.92,
        "max_spread": 0.06,
        "min_top_size": 10,           # contracts available at the quote
        "min_games_history": 6,       # player must have this many prior games
        "max_stake_per_trade": 25.0,
        "max_stake_per_player": 40.0,
        "max_stake_per_game": 100.0,
        "max_total_exposure": 0.5,    # fraction of bankroll at risk at once
        "max_open_positions": 30,
        "daily_loss_limit": 75.0,
        "max_drawdown": 0.25,         # from high-water mark -> circuit breaker
        "min_minutes_to_kickoff": 5,
        "skip_questionable": True,
    },
    "health": {
        "max_model_staleness_weeks": 3,
        "min_match_rate": 0.90,
        "max_median_disagreement": 0.10,
    },
    "combos": {
        "min_legs": 2, "max_legs": 6,
        "hold": 0.10,                 # assumed market-maker margin over the correlation-aware fair price
        "min_roi": 0.10,              # required expected return on a combo at the estimated quote
        "min_combo_p": 0.04,          # ignore lottery tickets below this joint probability
        "min_leg_p": 0.25, "max_leg_p": 0.93,
        "min_leg_edge": 0.0,          # leg must be at least as likely as the market says (log-odds)
        "max_leg_spread": 0.06,       # wider quotes have meaningless mids (RESEARCH.md)
        "allow_no": True,             # allow "under" legs (NO side of an N+ market)
        "same_game_only": False,
        "rungs_per_ladder": 4, "max_candidates": 70, "beam": 250, "draws": 40000,
        "max_stake_per_combo": 10.0,
        "exclude_stats": [],
        # Kalshi Help Center: receptions, receiving-yards and rushing-yards ladders can't be added to combos.
        # Empty this list if that changes; the exchange's own collection list is also checked (require_eligible).
        "not_combinable_stats": ["rec", "rec_yds", "rush_yds"],
        "require_eligible": True,
        "max_legs_per_game": 4,       # concentration limits
        "max_legs_per_player": 2,
        "max_pair_rho": 0.85,         # two legs this correlated are nearly the same bet
        "max_same_stat": 2,
        # policy (strategy.py / combo-run)
        "allowed_legs": [2, 3, 4, 5],
        "combo_kelly_fraction": 0.10,  # of robust Kelly; combos are far noisier than single props
        "max_weekly_combo_frac": 0.05, # total open combo stake as a fraction of bankroll
        "max_combos_per_run": 3,
        "policy_min_p": 0.10,         # don't bet long shots the backtest never validated (its combos hit 10-35%)
        "bt_candidates": 40,          # legs per week in the real-price backtest           # legs of one stat type: a model bias on that stat would sink them together
    },
}


def _deep_update(a: dict, b: dict) -> dict:
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


# Risk profiles (choose with `profile:` in config.yaml). Evidence: `python -m propbot tune-risk`, RESEARCH.md.
# balanced / aggressive use edge-realization trust by market-price band, fitted walk-forward. On 3,539 real-quote
# bets (2025-26, rules fixed in advance): ROI +7.7% (90% CI +2.5..+13.2%). Over 26 weeks from $500:
#   balanced   (0.20 Kelly) median +71%, 5th pct +14%, P(losing season) 2%, 95th-pct drawdown 22%
#   aggressive (0.50 Kelly) median +124%, 5th pct +12%, P(losing season) 3%, 95th-pct drawdown 37%
# (fills at the quoted price assumed; see RESEARCH.md).
PROFILES = {
    "conservative": {"model": {"trust": 1.0},
                     "risk": {"kelly_fraction": 0.10, "min_edge": 0.025, "max_disagreement": 0.20}},
    "balanced": {"model": {"trust": "auto"},
                 "risk": {"kelly_fraction": 0.20, "min_edge": 0.015, "max_disagreement": 0.15,
                          "max_stake_frac": 0.05, "max_total_exposure": 0.5, "max_open_positions": 100}},
    "aggressive": {"model": {"trust": "auto"},
                   "risk": {"kelly_fraction": 0.50, "min_edge": 0.015, "max_disagreement": 0.15,
                            "max_stake_frac": 0.08, "max_total_exposure": 0.8, "max_open_positions": 150,
                            "max_stake_per_player": 80.0, "max_stake_per_game": 200.0,
                            "daily_loss_limit": 150.0, "max_drawdown": 0.40}},
}


def load_config() -> dict:
    import copy
    cfg = copy.deepcopy(DEFAULTS)
    p = ROOT / "config.yaml"
    user = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
    prof = user.get("profile") or cfg.get("profile")
    if prof:
        _deep_update(cfg, copy.deepcopy(PROFILES[prof]))
    _deep_update(cfg, user)
    if cfg["risk"].get("max_stake_frac"):  # per-trade cap that scales with the bankroll
        cfg["risk"]["max_stake_per_trade"] = min(cfg["risk"].get("max_stake_per_trade", 1e9),
                                                 cfg["risk"]["max_stake_frac"] * cfg["risk"]["bankroll"]) \
            if "max_stake_per_trade" in user.get("risk", {}) else cfg["risk"]["max_stake_frac"] * cfg["risk"]["bankroll"]
    return cfg


def env(name: str, default: str | None = None) -> str | None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return os.environ.get(name, default)
