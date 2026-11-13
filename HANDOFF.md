# propbot — project handoff (state as of 2026-11-13)

## 2026-11-13: branches merged
`claude/peaceful-knuth-u3qs0s` now contains both lines of work:
- from `model-robustness`: opportunity / NGS / PFR features, depth charts, expert ranks, stage 1b spread model,
  cross-stat projections, hit rates, 3 bagged ladders, learned blend (off), research harness, two-season market board.
- from this branch: `context.py` (record, vs opponent/coach/former team, referee, schedule spot, travel),
  `defense.py` (expected opposing defenders, coverage/pass-rush quality, missing regulars, OL health, player vs
  defenders), `weather.py` (Open-Meteo forecast + reanalysis; context only, `model.use_weather` off),
  `combos.py` (copula correlations, Monte-Carlo pricing, leg/rung search, risk report, Kalshi eligibility),
  `report.py` (`matchup`), plus new `situational.py` (windows, half splits, game-script probabilities, opponent
  extras, availability, situational splits), `quotes.py` (scan quote log -> price movement / volatility /
  slippage) and `params.py` (250-parameter checklist -> PARAMETERS.md: 234 implemented, 16 need paid data).
- The new feature blocks have NOT been judged on the market board (`market-backtest` + `research --compare`).
  RESEARCH.md's rule is that a feature stays only if it helps there; do that before trusting them.
- Expert ranks: nflreadpy's github.com/.../raw URL is refused on some networks; `data._raw_download` now pulls
  the dynastyprocess files from raw.githubusercontent.com into data/external/.


Kalshi NFL player-prop model + trading bot. Python 3.13, venv at `.venv/`. Run everything as
`.venv/bin/python -m propbot <command>`. Work lives on branch `model-robustness`.

## What it does
1. Pulls free data into a parquet cache (`data/raw/`, no API key): nflverse player stats, PBP, snaps, injuries, rosters, schedules, depth charts, Next Gen Stats, PFR charting; ffverse opportunity stats; FantasyPros expert consensus ranks.
2. Builds one row per (skill player, game) with only pre-kickoff features (`propbot/features.py`, 387 features).
3. Three-stage LightGBM model (`propbot/model.py`):
   - Stage 1: regressor per stat → projection `mu` (plus helper targets: targets, snap share).
   - Stage 1b: regressor on the size of the miss → `sd`.
   - Stage 2: three monotone classifiers per stat, averaged; strike is a feature → P(stat > k) for any Kalshi "N+" rung. Inputs are out-of-fold stage-1 outputs (no leakage). Isotonic calibrators in `models/calibrators.joblib`.
4. Pulls open Kalshi markets (public API), parses + matches players (`markets.py`), prices them, blends with market mid (`market_weight` 0.7, in logit space), optional sportsbook re-leveling via The Odds API (`odds.py` — **written but never tested, no key**).
5. Risk engine (`risk.py`): unchanged. Edge after fee, price band, max disagreement, spread/size, history, skip Questionable, one rung per ladder, exposure caps, daily loss limit, drawdown breaker, kill switch file `KILL`. Fractional Kelly 0.20.
6. Executes paper (default) or live orders; SQLite ledger at `data/ledger.sqlite`.

## Files
- `config.py` paths, STATS, risk DEFAULTS; `config.yaml` overrides.
- `data.py` cache · `features.py` frame · `model.py` train/PropModel · `markets.py` Kalshi parsing · `kalshi.py` API client (now also the archive endpoints) · `edge.py` fees/EV/Kelly · `blend.py` learned model/market blend (off by default) · `risk.py` gates · `ledger.py` SQLite · `odds.py` sportsbook anchor · `bot.py` scan/run/settle/market backtest · `research.py` experiment harness · `cli.py` commands.
- `RESEARCH.md` — every experiment, what it was worth, what was dropped. Read it before changing the model.
- `models/`, `data/` git-ignored. `reports/backtest.json`, `reports/market_backtest.json` tracked.

## CLI
```
update            refresh current-season data
train [--refresh] [--no-backtest]   features + walk-forward backtest + fit final models (~30 min on 4 cores)
report            print backtest metrics
project [--player X] [--team KC]    upcoming-slate projections
scan [--top N] [--all] [--refresh]  price open Kalshi markets, place nothing
run [--live] [--loop MIN]           settle + scan + risk + orders (paper unless --live)
settle / status [--live]
market-backtest [--refresh]         model vs real Kalshi pre-kickoff quotes, 2025 week 5 onward
kill [--off] [--cancel]             kill switch
python -m propbot.research NAME key=value   run + score a model variant; --compare A B to compare saved runs
```

## Results
Models trained through **2026 week 9**.

Against real Kalshi quotes (47,731 tradeable settled markets, 2025 wk 9 – 2026 wk 9, all walk-forward):
- log-loss: market 0.4982, model 0.5033, **70/30 blend 0.4958**. Before this round the model was 0.5097 and the blend 0.4969 on the same markets.
- Replay of the default rules: 737 bets, ROI **+7.4%** (90% CI +0.1% … +14.8%). 2026 +10.1% (642 bets), 2025 −10.9% (95 bets). The old model on the same markets: +4.6% (−2.9% … +12.0%).
- A near-identical variant replayed at +12.9%. The replay is that noisy; trust the log-loss more than the ROI.

Last `scan` (2026-11-12, a Thursday): 2,956 open markets, 98% matched to players. Ledger is empty — **no paper or live trades placed yet.**

## Setup / safety
- `.env` (from `.env.example`): `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_LIVE_TRADING=0`, `ODDS_API_KEY`. Live orders need BOTH `KALSHI_LIVE_TRADING=1` AND `--live`. No `.env` exists. Nothing in this round touched an account; only public market data was read.
- Config unchanged: prod env, bankroll $500, kelly 0.20, min_edge $0.025/contract, max $25/trade, daily loss limit $75, market_weight 0.7.
- New machine: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m propbot train --refresh`, then `market-backtest` (first run downloads ~20k archived quotes, ~45 min, resumable).
- Old model files are not compatible with the new code. Retrain after pulling.

## Open ideas / next steps
- Paper trade (`run --loop 15`, Friday evening through kickoff) for a few weeks; then `settle`/`status`.
- Weather forecasts (temp/wind are excluded because the schedule file only has them after the game).
- Get an Odds API key and test `odds.py`.
- Log each scan's quotes to build a price history with real depth.
- Retrain weekly (`update` then `train`); `market-backtest --refresh` every few weeks.
