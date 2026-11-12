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
| Data | All free, no key. nflverse: weekly player stats, snap counts, play-by-play, schedules + closing lines, injury reports, rosters, depth charts, Next Gen Stats tracking, PFR charting. ffverse: opportunity-based expected stats. FantasyPros expert consensus ranks (dynastyprocess archive). 2015 → today, ~77k player-games. | |
| Features (387) | Rolling form at two half-lives, season/career means, volatility, snap share. Expected yards / receptions / TDs from the quality of each opportunity. Separation, air yards, rush yards over expected, pressure and drop rates. Target and carry share re-split among the teammates who are actually available. Depth chart rank. Weekly and rest-of-season expert ranks scraped before kickoff. Team pace and pass rate, opponent defense overall, vs. the position, and its pass rush. Starting-QB quality and QB changes, usage vacated by absent teammates, Vegas spread / total / implied totals, rest, injury status, age, draft capital. | Everything is computed from *prior* games or published before kickoff. |
| Stage 1 | LightGBM regressor per stat → projection `mu`. Also for two helper targets (targets, snap share). | The "expected yards" number. |
| Stage 1b | LightGBM regressor on the size of stage 1's miss → `sd`. | How wide this player's outcomes run. |
| Stage 2 | Three monotone LightGBM classifiers per stat (averaged) for `P(stat > k)` with the strike `k` as a feature, plus `mu`, `sd`, the projections of related stats, and how often the player cleared `k` in his last 5 and 12 games. | Prices every rung of a ladder from one model, and can never say 60+ is likelier than 50+. |
| Calibration | Isotonic regression fit on out-of-sample predictions. | So a 30% means 30%. |
| Blend | Final probability = 70% market price, 30% model, in log-odds. | The market sees news the model does not. |

Stage 2 is trained on stage-1 projections from models that never saw that season, so no
future information leaks in. This is gradient boosting, not deep learning — with ~77k rows of
tabular data it is the stronger choice. Training takes about 30 minutes on 4 cores.

`RESEARCH.md` has what was tried, what each change was worth, and what was thrown away.

## Results (be skeptical)

**Walk-forward, 2023–2026** — every season predicted by models trained only on earlier seasons:

| stat | log loss | naive rolling-avg baseline | AUC (tradeable strikes) | calibration error | projection MAE |
|---|---|---|---|---|---|
| pass_yds | 0.342 | 0.354 | 0.78 | 0.004 | 54.5 |
| pass_tds | 0.369 | 0.368 | 0.79 | 0.008 | 0.86 |
| pass_att | 0.442 | 0.463 | 0.76 | 0.011 | 5.99 |
| pass_comp | 0.387 | 0.400 | 0.78 | 0.006 | 4.25 |
| pass_int | 0.436 | 0.432 | 0.71 | 0.007 | 0.69 |
| rush_yds | 0.123 | 0.139 | 0.78 | 0.001 | 7.74 |
| rush_att | 0.119 | 0.146 | 0.78 | 0.001 | 1.30 |
| rec | 0.189 | 0.200 | 0.78 | 0.001 | 1.14 |
| rec_yds | 0.216 | 0.229 | 0.77 | 0.001 | 14.4 |
| rr_yds | 0.216 | 0.231 | 0.76 | 0.001 | 17.7 |
| td | 0.168 | 0.177 | 0.68 | 0.001 | 0.28 |

The model is well calibrated and clearly beats a rolling average everywhere except passing TDs
and interceptions, where it is no better.

**Against the real market** — 47,731 settled Kalshi prop markets with a tradeable quote, from
2025 week 9 through 2026 week 9, using the last bid/ask before kickoff:

| | log loss |
|---|---|
| Kalshi price | 0.4982 |
| model alone | 0.5033 |
| 70/30 blend | **0.4958** |

The market is more accurate than the model. The blend is more accurate than the market, in
both seasons, and that gap is about twice what it was before the new features.

Replaying the default rules with a flat $10 stake: **737 bets, +7.4% ROI after fees**, 90%
interval **+0.1% to +14.8%**. By season: 2026 +10.1% on 642 bets, 2025 −10.9% on 95 bets.

Read that carefully before risking money:

- The interval nearly touches zero, and one of the two seasons lost.
- Near-identical versions of this model replayed anywhere from +7% to +13%. The difference
  between them is noise, and so is most of any single number.
- The features were chosen while watching these same markets.
- Historical order-book depth is unknown; the replay assumes $10 always filled at the quote,
  and the quote is often the price at kickoff, slightly later than the bot would trade.
- History knows who actually played. Before a game the bot only knows the active roster and
  the injury report.
- "High accuracy" is not a meaningful target here. A 15¢ contract loses 85% of the time and
  can still be a great bet. What matters is calibration and edge versus price.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m propbot train        # downloads data, backtests, fits models (~30 min)
```

## Commands

```bash
python -m propbot update            # refresh this season's data (run after each week's games)
python -m propbot train             # rebuild features, walk-forward backtest, fit final models
python -m propbot report            # backtest metrics
python -m propbot market-backtest   # model vs real Kalshi pre-kickoff quotes, 2025 onward
python -m propbot market-backtest --refresh   # same, after downloading newly settled markets
python -m propbot.research NAME key=value     # try a model variant and score it (see RESEARCH.md)
python -m propbot project --team WAS   # projections for the upcoming slate
python -m propbot scan              # price every open prop market, show what passes the gates
python -m propbot run               # scan + risk engine + PAPER orders
python -m propbot run --loop 15     # same, every 15 minutes
python -m propbot settle            # settle finished positions
python -m propbot status            # bankroll, P&L, trades
python -m propbot kill              # kill switch (kill --off to release)
```

Weekly rhythm: `update` → `train --no-backtest` on Tuesday, then `run --loop 15` from Friday
evening through kickoff. Not earlier: the injury report and the weekly expert ranks, two of the
model's best inputs, are not out until Friday.

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
- No weather. The schedule file only has it after the game, so the model does not use it.
- The first `market-backtest` on a new machine downloads ~20,000 archived quotes one at a
  time (about 45 minutes, resumable).
- Pre-game only. No in-play pricing, no first-TD-scorer or longest-reception markets.
- "Protected" (`…PROT`) series are not traded.
- Rookies and players with fewer than 6 career games are skipped.
- `models/` and `data/` are git-ignored; run `train` after cloning.

Not financial advice. Trading event contracts can lose money.
