# propbot — project handoff (state as of 2026-11-12)

Kalshi NFL player-prop model + trading bot. Python 3.13, venv at `.venv/`. Run everything as
`.venv/bin/python -m propbot <command>`. Nothing is committed to git yet (repo has no commits).

## What it does
1. Pulls nflverse data (player stats, PBP, snaps, injuries, rosters, schedules) 2015–2026 into a parquet cache (`data/raw/`, no API key).
2. Builds one row per (skill player, game) with only pre-kickoff features (`propbot/features.py`, ~250 cols: rolling EWMs, team/opp pace & EPA, QB context, vacated targets/carries, injury flags, Vegas lines, weather).
3. Two-stage LightGBM model (`propbot/model.py`):
   - Stage 1: regressor per stat → projection `mu`.
   - Stage 2: one monotone classifier per stat, strike is a feature → P(stat > k) for any Kalshi "N+" rung. Trained on out-of-fold stage-1 projections (no leakage). Isotonic calibrators in `models/calibrators.joblib`.
4. Pulls open Kalshi markets (public API), parses + matches players (`markets.py`), prices them, blends with market mid (`market_weight` 0.7, in logit space), optional sportsbook re-leveling via The Odds API (`odds.py` — **written but never tested, no key**).
5. Risk engine (`risk.py`): edge after Kalshi quadratic fee, price band, max model/market disagreement, spread/size, history, skip Questionable, one rung per player ladder, per-player/game/total exposure caps, daily loss limit, drawdown breaker, kill switch file `KILL`. Fractional Kelly 0.20.
6. Executes paper (default) or live orders; SQLite ledger at `data/ledger.sqlite`.

## Stats covered (Kalshi series)
pass_yds KXNFLPASSYDS, pass_tds KXNFLPASSTDS, pass_att KXNFLPASSATT, pass_comp KXNFLPASSCOMP, pass_int KXNFLPASSINT,
rush_yds KXNFLRSHYDS, rush_att KXNFLRSHATT, rec KXNFLREC, rec_yds KXNFLRECYDS, rr_yds KXNFLRRYDS, td KXNFLTD.

## Files
- `propbot/config.py` — paths, STATS definitions/strike grids, risk DEFAULTS; `config.yaml` overrides them.
- `data.py` nflverse cache · `features.py` frame · `model.py` train/PropModel · `markets.py` Kalshi parsing · `kalshi.py` API v2 client (RSA-PSS/Ed25519 signing) · `edge.py` fees/EV/Kelly · `risk.py` gates · `ledger.py` SQLite · `odds.py` sportsbook anchor · `bot.py` scan/run/settle/market backtest · `cli.py` commands.
- `models/` trained models (gitignored), `reports/` backtests, `data/kalshi_settled.parquet` settled Kalshi quotes.

## CLI
```
update            refresh current-season nflverse data
train [--refresh] [--no-backtest]   features + walk-forward backtest + fit final models (~10 min)
report            print backtest metrics
project [--player X] [--team KC]    upcoming-slate projections
scan [--top N] [--all] [--refresh]  price open Kalshi markets, place nothing
run [--live] [--loop MIN]           settle + scan + risk + orders (paper unless --live)
settle / status [--live]
market-backtest [--min-volume 25]   model vs real Kalshi pre-kickoff quotes
kill [--off] [--cancel]             kill switch
```

## Results so far
Models trained through **2026 week 9**. Walk-forward backtest (2023–2026): model beats naive logloss on every stat except pass_tds and pass_int (roughly tied); tradeable-strike AUC 0.67 (td) – 0.80; calibration error ≤ 0.011.

Market backtest vs real Kalshi pre-kickoff quotes (2026 wk 1–9, 41,971 markets, 136 games):
- logloss model 0.4769, market 0.4662, **blend 0.4648** (model alone is worse than market; the blend beats it).
- Strategy: 801 bets, 53.6% win, ROI **+9.0%** (90% CI +0.7% … +17.5%), P&L +$700 on $7.8k staked.
- Good: rec (+19%, 239 bets), rush_att (+23%), td (+40%, 35 bets). Bad: rec_yds (−4.8%, 221 bets), rr_yds (−28%), pass_comp (−45%, small n).

Last `scan` (2026-11-12): 2,783 markets priced, 16 passed all gates. Ledger is empty — **no paper or live trades placed yet.**

## Setup / safety
- `.env` (from `.env.example`): `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_LIVE_TRADING=0`, `ODDS_API_KEY`. Live orders need BOTH `KALSHI_LIVE_TRADING=1` AND `--live`. No `.env` exists yet.
- Current config: prod env, bankroll $500, kelly 0.20, min_edge $0.025/contract, max $25/trade, daily loss limit $75.
- New machine: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m propbot train --refresh` (data/ and models/ are gitignored, so regenerate or copy them).

## Open ideas / next steps
- Paper trade (`run --loop 15`) for a few weeks before going live; then `settle`/`status`.
- Consider disabling or raising thresholds for rr_yds, rec_yds, pass_comp (negative backtest ROI).
- Get an Odds API key and actually test `odds.py`.
- Retrain weekly (`update` then `train`) as new weeks come in.
- Make an initial git commit.
