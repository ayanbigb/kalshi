# propbot — Kalshi NFL player-prop model and bot

Prices Kalshi's NFL player-prop markets with a model trained on 11 seasons of player data,
compares its fair value to the live order book, and (after a stack of risk gates) trades the
gap. Paper mode by default; live trading is behind two explicit switches.

## How Kalshi player props work

Each prop is a ladder of binary "N+" contracts per player per game, e.g.
`KXNFLRECYDS-26NOV15BUFNYJ-BUFKSHAKIR10-40` = *Khalil Shakir 40+ receiving yards, BUF @ NYJ*.
A YES contract pays $1 if it happens. Price ≈ implied probability. Taker fee is
`ceil(0.07 × contracts × price × (1 − price))`. If the player is active but never plays a snap,
the market settles at the pre-game fair price instead of YES/NO.

Series covered: passing yards / TDs / attempts / completions / INTs, rushing yards / attempts,
receptions, receiving yards, rush+rec yards, and touchdowns (1+, 2+, 3+).

## How the model works

| Stage | What | Why |
|---|---|---|
| Data | nflverse (free, no key): weekly player stats, snap counts, play-by-play, schedules + closing lines, injury reports, rosters. 2015 → today, ~77k player-games. | |
| Features (~200) | Rolling form at two half-lives, season/career means, volatility, snap share, target/air-yard/carry share, red-zone and goal-line usage, team pace and pass rate over expected, opponent defense overall and vs. the player's position, starting-QB quality and QB changes, usage vacated by absent teammates, Vegas spread / total / implied team total, weather, rest, injury status, age, draft capital. | Everything is computed from *prior* games only. |
| Stage 1 | LightGBM regressor per stat → projection `mu`. | The "expected yards" number. |
| Stage 2 | One monotone LightGBM classifier per stat for `P(stat > k)` with the strike `k` as a feature. | Prices every rung of a ladder from one model, and can never say 60+ is likelier than 50+. |
| Calibration | Isotonic regression fit on out-of-sample predictions. | So a 30% means 30%. |

Stage 2 is trained on stage-1 projections from models that never saw that season, so no
future information leaks in. This is gradient boosting, not deep learning — with ~77k rows of
tabular data it is the stronger choice, and it trains in 10 minutes on a laptop.

## Results (be skeptical)

**Walk-forward, 2023–2026** — every season predicted by models trained only on earlier seasons:

| stat | log loss | naive rolling-avg baseline | AUC (tradeable strikes) | calibration error |
|---|---|---|---|---|
| pass_yds | 0.350 | 0.354 | 0.79 | 0.007 |
| pass_tds | 0.369 | 0.368 | 0.80 | 0.007 |
| pass_att | 0.461 | 0.463 | 0.77 | 0.005 |
| pass_comp | 0.397 | 0.400 | 0.78 | 0.007 |
| pass_int | 0.434 | 0.432 | 0.71 | 0.011 |
| rush_yds | 0.125 | 0.139 | 0.78 | 0.001 |
| rush_att | 0.125 | 0.146 | 0.79 | 0.001 |
| rec | 0.191 | 0.200 | 0.78 | 0.001 |
| rec_yds | 0.218 | 0.229 | 0.77 | 0.001 |
| rr_yds | 0.219 | 0.231 | 0.77 | 0.001 |
| td | 0.170 | 0.177 | 0.67 | 0.002 |

The model is well calibrated and clearly beats a rolling average for rushing and receiving.
For QB passing TDs and interceptions it is no better than the baseline.

**Against the real market** — 41,971 settled Kalshi prop markets from weeks 1–9 of 2026
(136 games), using the last bid/ask quoted before kickoff:

- Log loss: market **0.4662**, model alone **0.4769**, blend **0.4648**.
  The market is more accurate than the model. The model adds a little on top of it.
- Replaying the default rules with a flat $10 stake: **801 bets, +9.0% ROI after fees**,
  90% interval **+0.7% to +17.5%**, 6 of 9 weeks profitable.

Read that last line carefully before risking money:

- It is nine weeks. The interval nearly touches zero.
- `market_weight` and `min_edge` were set after looking at this same data.
- Historical order-book depth is unknown; the replay assumes $10 always filled at the quote.
- Features use closing lines and actual inactives, i.e. information as of roughly kickoff.
- "High accuracy" is not a meaningful target here. A 15¢ contract loses 85% of the time and
  can still be a great bet. What matters is calibration and edge versus price.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m propbot train        # downloads data, backtests, fits models (~10 min)
```

## Commands

```bash
python -m propbot update            # refresh this season's data (run after each week's games)
python -m propbot train             # rebuild features, walk-forward backtest, fit final models
python -m propbot report            # backtest metrics
python -m propbot market-backtest   # model vs real Kalshi pre-kickoff quotes this season
python -m propbot project --team WAS   # projections for the upcoming slate
python -m propbot scan              # price every open prop market, show what passes the gates
python -m propbot run               # scan + risk engine + PAPER orders
python -m propbot run --loop 15     # same, every 15 minutes
python -m propbot settle            # settle finished positions
python -m propbot status            # bankroll, P&L, trades
python -m propbot kill              # kill switch (kill --off to release)
```

Weekly rhythm: `update` → `train --no-backtest` on Tuesday, then `run --loop 15` from Friday
(after injury reports) through kickoff.

## Risk engine

Six independent gates, all must pass (`propbot/risk.py`, thresholds in `config.yaml`):

1. **Edge** — expected profit after fees ≥ `min_edge` and `min_roi`.
2. **Sanity** — price in the tradeable band; if model and market disagree by more than
   `max_disagreement`, assume the market knows something (injury, role change) and skip.
3. **Liquidity** — two-sided quote, tight spread, enough size at the touch.
4. **Data** — player matched, enough game history, not Questionable, game not about to start.
5. **Portfolio** — one rung per player ladder, caps per trade / player / game / total exposure.
6. **Breakers** — kill switch, daily loss limit, max drawdown.

Sizing is fractional Kelly (20%) on a probability that is already shrunk 70% toward the market.

## API keys

| Key | Needed for | Where |
|---|---|---|
| none | data, training, scanning, paper trading | — |
| `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH` | live orders, real balance | kalshi.com → Account & security → API keys |
| `ODDS_API_KEY` (optional) | sportsbook prop lines as an anchor | the-odds-api.com |

Copy `.env.example` to `.env`. Never commit `.env` or the key file (both are git-ignored).

## Going live

Live orders need **both** `KALSHI_LIVE_TRADING=1` in `.env` **and** `--live` on the command.
Before that:

1. Paper trade for several weeks and compare `status` to the backtest.
2. Set `kalshi.env: demo` with demo-environment keys and place a test order. **The live order
   path (`Kalshi.buy`, V2 `/portfolio/events/orders`) is written to the published spec but has
   never been executed.** Verify YES/NO side mapping on demo first.
3. Start with small caps in `config.yaml`.

## Known gaps

- Sportsbook anchor (`propbot/odds.py`) is untested — no key was available when it was written.
- No weather forecast for upcoming outdoor games (temp/wind are blank until played).
- Pre-game only. No in-play pricing, no first-TD-scorer or longest-reception markets.
- "Protected" (`…PROT`) series are not traded.
- Rookies and players with fewer than 6 career games are skipped.
- `models/` and `data/` are git-ignored; run `train` after cloning.

Not financial advice. Trading event contracts can lose money.
