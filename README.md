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
python -m propbot matchup "Smith-Njigba"   # every factor for one player's next game
python -m propbot combos            # multi-leg combos: how many legs, which rung, full risk report
python -m propbot combos --same-game --team SEA --max-legs 4
python -m propbot combo-backtest    # correlation model vs "legs are independent" on an unseen season
python -m propbot params            # 250-parameter checklist coverage -> PARAMETERS.md
python -m propbot weather-backfill  # historical game weather (slow, resumable)
python -m propbot combo-math --p-leg 0.6 --edge 0.05 --hold 0.10   # 3 vs 5 legs: EV, variance, Kelly, growth
python -m propbot combo-backtest-market [--all-stats]  # combos by leg count on real 2025-26 prices
python -m propbot combo-plan --bankroll 500 --weeks 8 --target 0.2  # stakes, returns, drawdowns, ruin
python -m propbot combo-run         # policy: choose, size, record PAPER combos (combo-settle to settle)
python -m propbot run --loop 15 --combos   # singles + combos each pass
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

## Matchup and context features

Added on top of the features above (all computed from games before kickoff):

| Group | Features | Module |
|---|---|---|
| History vs opponent | prior games vs this franchise, and his average over/under-performance vs his own rolling average in them (rec yds, rec, rush yds, carries, targets, pass yds, TDs), shrunk toward 0 for small samples; playing a former team | `context.py` |
| History vs coach | same split against the opposing head coach (follows coaches across teams); own head coach new this season | `context.py` |
| Opposing defenders | the defenders expected to play (snap counts + roster + injury report), weighted by expected snap share: CB / S / LB yards-per-target and completion % allowed, top corner, DB TD rate, front pressures per game, missed-tackle rate (PFR advanced defense, 2018+); snap share of regular CB / S / LB / DL who are missing | `defense.py` |
| Player vs those defenders | his over/under-performance in previous games where each of tonight's defenders was on the field (≥30% of snaps), weighted by how much each will play — DBs for receiving, front for rushing | `defense.py` |
| Own line | snap share of regular offensive linemen missing | `defense.py` |
| Team record | season-to-date win% (shrunk), point differential per game, last-3 differential, streak — both teams | `context.py` |
| Weather | temperature, sustained wind, gusts, rain, snow, humidity for kickoff → +3h at the venue (Open-Meteo forecast for upcoming games, reanalysis for history). Shown in `matchup` and used by the combo engine; kept out of the model by default (`model.use_weather`) because the free forecast feed is not always reachable at scan time | `weather.py` |
| Spot | primetime, early window, Thursday, short week, rest edge vs opponent, travel distance, time-zone shift, west-coast-team-early-kickoff, altitude, referee crew's history vs the closing total | `context.py` |

| Windows and splits | last-3/5/10 averages, adjusted yards per attempt, yards per reception, touches per snap, 5-game usage trend, first/second-half yards; the player's past over/under-performance at home/away, in divisional, primetime, cold and bad-weather games, vs top-10/bottom-10 defenses, after a loss, on short rest, by game script, with the current QB, with a top teammate out, and over the last 3 meetings with this opponent | `situational.py` |
| Game script | de-vigged moneyline win probability, P(blowout), P(one-score game), P(near-tie finish), expected plays | `situational.py` |
| Opponent extras | 3rd-down conversion, red-zone TD rate and sack rate allowed; yards allowed to the opponent's WR1, WR2, TE1, RB1 | `situational.py` |
| Availability | games missed this season, weeks on the injury report in the last 17, games since returning, consecutive road games | `situational.py` |

`PARAMETERS.md` maps a 250-item prop/combo checklist to these columns (`python -m propbot params`):
234 are implemented as model features, proxies, combo-engine outputs or market fields; 16 need paid
data (routes run, coverage charting, DVOA, block/route win rates) or drive ids and are marked so.

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

**Risk report** for each shortlisted combo: P(all hit) with a calibration band (every leg moved by twice
its stat's historical calibration error) and Monte-Carlo error, P(≥1 leg fails), P(≥2 fail), P(all
fail), the same combo if legs were independent, a pessimistic probability with legs shrunk 90% toward the
market, every pair's correlation and P(B | A), games / same-team / opposing-team counts, and a season
simulation of betting the shortlist weekly (expected P&L, P(losing season), max drawdown). Searches are
capped at 4 legs per game, 2 per player and |correlation| ≤ 0.85 between any two legs.

**Kalshi rules.** Only markets the combo builder accepts are used: the exchange's own NFL combo collection
list (yes-only events get no NO legs), minus `combos.not_combinable_stats`, which defaults to receptions,
receiving yards and rushing yards per the Help Center. Protected (`…PROT`) markets are never used.
Player stats settle on official stats including overtime, which is what nflverse records too.

**The math and the evidence** (edge realization, real-price combo backtest by leg count, 3-vs-5-leg
analysis, staking plan, bot policy) are in `STRATEGY.md`.

Combo prices come from market makers on request (RFQ), and Kalshi doesn't publish them, so `hold` in
`config.yaml` is an assumption. Build the combo in the app and compare the quote to "accept up to".
The edge of a combo is mostly the product of its legs' edges, so it is only as good as the leg model;
more legs compound both the edge and the margin.

## Known gaps

- Sportsbook anchor (`propbot/odds.py`) is untested — no key was available when it was written.
- Weather is not a model input by default (the free forecast quota is shared and sometimes gone);
  it is shown in `matchup` and used for combo context.
- Combo quotes come from market makers on request; `combos.hold` is an assumption.
- The first `market-backtest` on a new machine downloads ~20,000 archived quotes one at a
  time (about 45 minutes, resumable).
- Pre-game only. No in-play pricing, no first-TD-scorer or longest-reception markets.
- "Protected" (`…PROT`) series are not traded.
- Rookies and players with fewer than 6 career games are skipped.
- `models/` and `data/` are git-ignored; run `train` after cloning.

Not financial advice. Trading event contracts can lose money.
