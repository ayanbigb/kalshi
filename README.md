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
python -m propbot matchup "Smith-Njigba"   # every factor for one player's next game
python -m propbot combos            # suggest multi-leg combos: how many legs, which rung of each ladder
python -m propbot combos --same-game --team SEA --max-legs 4
python -m propbot combo-backtest    # does the correlation model beat "legs are independent"?
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

## Matchup and context features

Added on top of the original ~250 columns (all computed from games before kickoff):

| Group | Features | Module |
|---|---|---|
| History vs opponent | prior games vs this franchise, and his average over/under-performance vs his own rolling average in them (rec yds, rec, rush yds, carries, targets, pass yds, TDs), shrunk toward 0 for small samples; playing a former team | `context.py` |
| History vs coach | same split against the opposing head coach (follows coaches across teams); own head coach new this season | `context.py` |
| Opposing defenders | the defenders expected to play (snap counts + roster + injury report), weighted by expected snap share: CB / S / LB yards-per-target and completion % allowed, top corner, DB TD rate, front pressures per game, missed-tackle rate (PFR advanced defense, 2018+); snap share of regular CB / S / LB / DL who are missing | `defense.py` |
| Player vs those defenders | his over/under-performance in previous games where each of tonight's defenders was on the field (≥30% of snaps), weighted by how much each will play — DBs for receiving, front for rushing | `defense.py` |
| Own line | snap share of regular offensive linemen missing | `defense.py` |
| Team record | season-to-date win% (shrunk), point differential per game, last-3 differential, streak — both teams | `context.py` |
| Weather | temperature, sustained wind, gusts, rain, snow, humidity for kickoff → +3h at the venue. History from Open-Meteo reanalysis (so the model can learn the effect), upcoming games from the Open-Meteo forecast | `weather.py` |
| Spot | primetime, early window, Thursday, short week, rest edge vs opponent, travel distance, time-zone shift, west-coast-team-early-kickoff, altitude, referee crew's history vs the closing total | `context.py` |

`matchup` prints all of it for one player, including the list of expected opposing defenders and how he has
done against each.

Limits: free data doesn't say which corner covered which receiver on a play, so "vs defender" means
"in games where that defender played". For upcoming games the injury report is all that's known about
absences (history knows game-day inactives), so scans after Friday's report are more accurate; a
Questionable defender counts as half-missing.

## Combos

`combos` builds Kalshi multi-leg combos (every leg must hit) from the priced scan.

1. **Correlation model.** Each (player, stat) in a game gets a latent normal variable; all rungs of one
   ladder share it, so "JSN 80+" can only hit if "JSN 60+" does. Correlations are estimated from
   out-of-sample residuals (actual − projection, as normal scores) for each pair of roles — same player
   (rec ↔ rec yds), teammates (QB pass yds ↔ WR rec yds, RB1 ↔ RB2), opponents (shootouts) — and shrunk
   toward a coarser estimate when the sample is small (`models/copula.json`, fitted by `train`).
2. **Pricing.** One Monte-Carlo simulation of the slate (40k draws). For every combo:
   `P(model)` uses our leg probabilities; `fair` uses the market's own leg mids with the same correlations
   (what a market maker who prices correlation would charge before margin); `independent` is the naive
   product. The expected quote is `fair × (1 + hold)`; **accept up to** is the highest price that still
   leaves `min_roi` of expected return.
3. **Search.** Beam search over YES (over) and NO (under) legs, choosing how many legs and which rung of
   each ladder, ranked by expected log-bankroll growth at fractional Kelly. Output: the best combo for
   each leg count, a diverse shortlist, and for the top combos what swapping each leg to another rung
   does (e.g. 60+ vs 80+).

On games it never saw (2026), the correlation model prices random same-game combos better than treating
legs as independent (log loss 0.346 → 0.341 for 3 legs, 0.214 → 0.209 for 4; `combo-backtest`).
Legs must pass the same model/market disagreement gate as single bets.

Combo prices come from market makers on request (RFQ), and Kalshi doesn't publish them, so `hold` in
`config.yaml` is an assumption. Build the combo in the app and compare the quote to "accept up to".
The edge of a combo is mostly the product of its legs' edges, so it is only as good as the leg model;
more legs compound both the edge and the margin.

## Known gaps

- Sportsbook anchor (`propbot/odds.py`) is untested — no key was available when it was written.
- Pre-game only. No in-play pricing, no first-TD-scorer or longest-reception markets.
- "Protected" (`…PROT`) series are not traded.
- Rookies and players with fewer than 6 career games are skipped.
- `models/` and `data/` are git-ignored; run `train` after cloning.

Not financial advice. Trading event contracts can lose money.
