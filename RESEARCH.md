# Research log

What was tried to make the model better, what it did, and what was thrown away. Run any of it
again with `python -m propbot.research NAME key=value ...` (keys are `model.OPTS`); compare saved
runs with `python -m propbot.research --compare A B`.

## How a change is judged

Two scoreboards, always the same:

- **Grid**: log-loss over every strike in each stat's training grid, 2023–2026, walk-forward
  (a season is predicted by models fitted on earlier seasons only). Big sample, but most of it
  is strikes nobody trades.
- **Market**: only strikes Kalshi actually listed, scored against the real last quote before
  kickoff. 48,529 tradeable markets (spread ≤ 6¢) from 2025 week 5 through 2026 week 9, over
  330 games. This is the one that matters.

On the market board the headline is not "is the model more accurate than Kalshi" (it is not) but
**how much a 70/30 market/model blend beats the market alone**, with a 5%–95% interval from
resampling whole games. An edge can only come from there.

Run-to-run noise on the market board is about ±0.001 log-loss. Differences smaller than that are
not findings.

## Results

Log-loss, lower is better. "Blend gain" is market minus blend, ×1000, higher is better.

| version | what changed | grid | market: model alone | blend gain [5%, 95%] |
|---|---|---|---|---|
| Kalshi price | | | 0.5004 | |
| baseline | the model as handed over | 0.2786 | 0.5118 | +1.28 [+0.12, +2.46] |
| v2 | opportunity, tracking and charting features; spread model; cross-stat projections; hit rates | 0.2733 | 0.5078 | +2.16 [+1.21, +3.20] |
| v3 | + depth chart rank | 0.2730 | 0.5065 | +2.51 [+1.54, +3.59] |
| v4 | + expert consensus ranks | 0.2729 | 0.5050 | +2.71 [+1.70, +3.71] |
| v4, 3 bags | ladder = average of 3 seeds | 0.2723 | 0.5049 | +2.61 [+1.61, +3.61] |
| v4, no weather | temp and wind removed | 0.2736 | 0.5059 | +2.31 [+1.32, +3.33] |
| **shipped** | v4, 3 bags, no weather | 0.2733 | 0.5054 | +2.37 [+1.40, +3.35] |

The gap between the model and Kalshi's own price went from 0.0114 to 0.0050. The blend's edge
over the market roughly doubled and its interval no longer touches zero in either season.

Weather is out even though it scored slightly better; see "Train/serve mismatches found".

## What was added, and why

**Opportunity instead of outcome** (ffverse `ff_opportunity`). Expected receptions, yards and
TDs given where and how a player was targeted or handed the ball. A 9-target, 40-yard game and
a 3-target, 70-yard game predict next week very differently; raw yards cannot tell them apart.

**Tracking and charting** (Next Gen Stats 2016+, PFR 2018+). Separation, cushion, intended air
yards, rush yards over expected, stacked boxes, time to throw, pressure and bad-throw rates,
drops, yards before and after contact, and the opposing defense's pressure, blitz and missed
tackles.

**More from play-by-play.** Completion-probability-weighted targets, explosive plays, stuffs,
third-down targets, scrambles and kneel-downs (they count as rush attempts), and for teams:
shotgun and no-huddle rate, depth of target, explosive rate, QB hits, red-zone pass rate.

**Who is actually available.** A player's target and carry share is renormalised over the
teammates who are in the lineup, then multiplied by team volume (`exp_targets`, `exp_carries`),
with rank within position by snaps, targets and carries.

**Depth charts.** Where the team lists the player going into the game.

**Expert consensus ranks** (FantasyPros via the dynastyprocess archive, 2020+). The weekly
positional rank is published on Friday, so it already knows about the week's injuries and role
changes; the rest-of-season rank is a slower read on role that also exists before week 1. Only
ranks scraped on an earlier day than kickoff are used. This was the single most useful
addition, and it helped most where the first error breakdown said the model was worst: weeks
1–2, rookies, and players whose role had just changed.

**Stage 1b, cross-stat projections, hit rates.** The ladder model now also sees a projected
spread (expected absolute miss), the projections for related stats and for two helper targets
(targets, snap share), and how often the player cleared this exact strike in his last 5 and 12
games. Alone each is worth almost nothing (see below); together, on top of the v4 features,
they are worth about 0.0015 on the market board.

**Kalshi history.** The archive endpoints are now read, which added the 2025 season (20,440
settled markets with quotes) to the 2026 ones. Everything about the market is now judged on
two seasons instead of nine weeks.

## 2026-11-13: the matchup / situational parameter set (merged branch)

~150 more inputs from a 250-item prop checklist (`PARAMETERS.md`): player vs this opponent, coach,
former team and the specific defenders he will face; opposing coverage and pass-rush quality and missing
regulars; own OL health; team records; last-3/5/10 windows, half splits, usage trends; game-script
probabilities; opponent 3rd-down / red-zone / role-specific yards allowed; availability history; and
situational splits (home/away, division, primetime, cold, bad weather, defense tier, after a loss, short
rest, script, with this QB, teammate out). Weather stays out of the model (context only).

| version | grid | market: model alone | blend | blend gain |
|---|---|---|---|---|
| shipped (model-robustness) | 0.2733 | 0.5033 | 0.4958 | +2.32 |
| + matchup / situational set | 0.2735 | 0.5046 | 0.4960 | +2.11 |

No gain; the model alone is slightly worse (0.0013, just outside noise) and the blend is equal within
noise.

Ablation, all three trained here on the same frame and scored on the same 48,529 markets:

| variant | grid | market: model alone | blend gain [5%, 95%] |
|---|---|---|---|
| without the set (= shipped model) | 0.2733 | 0.5054 | +2.37 [+1.40, +3.35] |
| structural part only (no history splits / vs-opponent / vs-defender residuals) | 0.2730 | 0.5062 | +2.33 [+1.29, +3.37] |
| whole set | 0.2731 | 0.5064 | +2.27 [+1.31, +3.27] |

Slightly better on the grid, slightly worse on the market board, in both forms. Decision: the price model
is trained without the set (`model.drop_feats` in config.py); the columns are still built for `matchup`, the
combo engine and PARAMETERS.md. Likely reason: the market already prices matchups, records and splits, so
they add nothing the blend can use, while the noisy history splits cost a little accuracy. Replay: 781 bets, +12.4% (90%: +4.8% … +20.1%) vs +7.4% for the shipped model on the same
markets: replay noise, not evidence. The columns stay in the frame because `matchup`, the combo engine
and `PARAMETERS.md` use them. To train without them (the ablation that would restore the shipped model):

    python -m propbot.research no_matchup drop_feats='^(sp_|vs|opp_u_|ol_miss|rec_(gp|wpct|pd_pg|pd_l3|streak)|opp_rec_|wpct_diff|ref_tot|travel_km|tz_shift|body_clock|altitude|primetime|early_window|thursday|short_week|rest_edge|hc_new|aya_|ypr_e6|touch_per|h1_|h2_|ml_win|p_blowout|p_close|p_ot|p_trail|exp_plays|opp_d3conv|opp_drz_td|opp_dsack_rate|opp_alw_(wr|te|rb|my)|road_streak|games_missed|games_since|inj_hist|role_rank)|_m(3|5|10)$|_trend5$'

The shipped `models/` are trained WITHOUT them (config `model.drop_feats`).

## 2026-11-13: how much risk the singles strategy can take (tuning.py, `tune-risk`)

Edge realization says the market under-weights the blend's disagreement (lambda ~1.2-1.9 depending on the
quote set). Three ways of using that, replayed on 2025-26 real quotes with fees, bets fixed in advance:

| rules | bets | ROI per bet | 90% CI | 2025 | 2026 |
|---|---|---|---|---|---|
| shipped (blend, min_edge 0.025, max_dis 0.25) | 1,008 | +6.4% | +0.4% … +11.8% | −10.1% | +10.2% |
| rules re-picked each week from a grid (walk-forward) | 3,595 | +4.8% | | | |
| trust = lambda fitted on earlier weeks only, shipped gates | 3,556 | +5.6% | +2.7% … +8.8% | +4.2% | +6.1% |
| trust = lambda walk-forward, min_edge 0.015, max_dis 0.15 | 3,558 | +4.8% | +0.5% … +9.4% | +7.7% | +3.9% |

Grid re-picking was unstable (it flipped between trust 1.75 and 2.5) and is not used. The lambda-trust
rules have no free knob and are positive in both seasons. Bankroll replay of the last row (26 weeks, $500,
2,000 week-resampled seasons, fills at the quote assumed):

| Kelly | median return | 5th pct | P(losing season) | 95th-pct drawdown |
|---|---|---|---|---|
| 0.10 | +30% | +5% | 2% | 10% |
| 0.20 | +54% | +9% | 2% | 19% |
| 0.35 | +53% | −3% | 6% | 28% |
| 0.50 | +74% | −6% | 7% | 37% |
| 0.75 | +94% | −10% | 8% | 45% |

**Trust by price band.** Splitting those bets by price showed YES bets at 30–70¢ losing (50–70¢: won 54.5%
vs 64.9% predicted, ~3 SE). Edge realization by market price: 0–15% 1.99, 15–30% 1.66, 30–50% 1.69,
50–70% **1.30**, 70–85% 1.70, 85–100% 1.13; the efficient middle of a ladder realizes the least. Trust fitted
per band (walk-forward, shrunk toward the pooled value with 3,000 pseudo-rows):

| trust | bets | ROI | 90% CI | 2025 | 2026 | YES @ 30–70¢ |
|---|---|---|---|---|---|---|
| one value | 3,558 | +4.8% | +0.4% … +9.4% | +7.7% | +3.9% | −5.6% |
| by price band | 3,539 | **+7.7%** | **+2.5% … +13.2%** | +11.5% | +6.6% | −0.5% |

| Kelly (banded) | median 26-wk return | 5th pct | P(losing season) | 95th-pct drawdown |
|---|---|---|---|---|
| 0.10 | +40% | +12% | 1% | 11% |
| 0.20 | +71% | +14% | 2% | 22% |
| 0.35 | +88% | +11% | 3% | 28% |
| 0.50 | +124% | +12% | 3% | 37% |
| 0.75 | +174% | +15% | 3% | 43% |

Shipped as profiles (config.py PROFILES): `balanced` = 0.20 Kelly, `aggressive` = 0.50 Kelly (default in
config.yaml), both `trust: auto` (banded, models/edge_realization.json), min_edge 0.015, max_dis 0.15, stake caps
scaled to the bankroll. 0.75 Kelly pushes the 95th-pct drawdown past 40% for the gain; not offered.
Caveats: the replay assumes every bet fills at the last quote (live depth is now checked with
`Kalshi.fill_price`), and ~135 bets a week is a lot of positions to manage.

## 2026-11-13: Kalshi price paths (does a line move say whether the model is right?)

Hourly prices for 34,700 settled 2026 markets (the 30 hours before kickoff). Edge realization split by how
the market moved over the last ~24h: toward the model 1.59, away from it 1.55, barely moved 1.18. And the move
itself has no predictive power beyond the final price (lambda on the move: −0.06). The market prices its own
momentum; the bot neither chases nor fades line moves. Not used.

## 2026-11-13: FTN charting and participation (charting.py)

New free data: FTN charting 2022+ (first-read targets, catchable / contested / created receptions, screens,
play-action, out-of-pocket, interception-worthy and throwaway rates, box counts, RPOs, blitz and pass-rusher
counts) as rolling features, and nflverse participation 2016-2025 (routes, man/zone) as season-lagged traits
only (route participation, targets and yards per route run, yards per target vs man and vs zone, opponent man
rate last season), because participation is published after the season and would be stale live.

| variant | grid | market: model alone | blend gain |
|---|---|---|---|
| shipped | 0.2733 | 0.5054 | +2.37 [+1.40, +3.35] |
| + charting / participation | 0.2735 | 0.5056 | +2.39 [+1.39, +3.40] |

A tie. Better for rush yds, rec, td, pass TDs; worse for pass yds and INTs. Not adopted (picking per stat after
seeing this would be fitting the test set); the columns are built and shown in `matchup`.

## 2026-11-14: real combo markups (from the connected account, read-only)

22 combos bought on the account consist only of NFL player-prop legs that have pre-kickoff quotes in the
settled-market cache. Price paid vs the correlation-aware fair value of the legs (copula, leg mids):
5–9 legs median +5% (−4% to +14%), 10–13 legs about +27%, 14+ legs +72% to +2,500%. Versus the plain product
of leg prices the median was +34%, because same-game legs are correlated. The combo engine now uses
`hold_by_legs` 6% / 27% / 75% instead of a flat 10%. The same combos contain 135 rushing-yards, 62
receiving-yards and 61 receptions legs, so those stats are combinable; `not_combinable_stats` is now empty.
At the prices paid, the model's median EV was −14% (5–9 legs), −32% (10–13), −50% (14+).

## What did not work

- **Learned model/market blend** (`propbot/blend.py`). A ridge logistic that starts at the
  70/30 mix and learns a weight per stat. Walk-forward it was worse than the fixed weight at
  every amount of shrinkage tried, and only matched it when shrunk so hard it *was* the fixed
  weight. In-sample it says the model deserves ~0.4; out of sample that does not hold. It is
  still computed and reported by `market-backtest`, and is off by default
  (`model.learned_blend`).
- **Training the ladder only on market-like rows** (`focus=True`). Kalshi lists players with a
  real role, and the history is mostly bit-part players, so this looked promising. It made the
  market board slightly worse (0.5038 vs 0.5023 at the time).
- **Rank shares among position mates and the QB's expert rank** (v5). No better than v4.
- **Refitting mid-season in the backtest** (`split_weeks=(1,10)`). Better on the grid (0.2723),
  no better on the market board, twice the compute.
- **60 context features for the ladder instead of 36.** No change.
- **Dropping any one of spread / cross-stat projections / hit rates from v2.** No change, which
  is why they are described above as jointly useful rather than individually.
- **Level vs shape.** Splitting the model-market disagreement into "the whole ladder is shifted"
  and "this rung is off" and weighting them separately: same weights, no gain.

## Things learned about the market itself

- Kalshi's prop prices are well calibrated. There is no blanket favourite/longshot tilt big
  enough to trade after the spread and fee.
- The markets are liquid: median spread 1¢, median volume ~4,000 contracts. Pass attempts,
  completions, rush attempts and rush+rec yards are the exception (about a fifth of their
  quotes are wider than 6¢), and a wide quote's mid is meaningless. The first version of the market
  board counted those and made the model look far better than it is on those four stats.
- The best single market weight is flat between 0.6 and 0.7.

## The trading replay, and how much to trust it

Default rules, flat $10 stake, best rung per player/stat/game, fees included, every price
walk-forward. Same 47,731 markets for both rows.

| model | bets | ROI | 90% interval | 2025 | 2026 |
|---|---|---|---|---|---|
| baseline | 886 | +4.6% | −2.9% … +12.0% | +1.7% (112) | +5.1% (774) |
| v4 | 855 | +12.9% | +5.7% … +19.9% | +7.9% (110) | +13.6% (745) |
| shipped | 737 | +7.4% | +0.1% … +14.8% | −10.9% (95) | +10.1% (642) |

v4 and the shipped model are almost the same model (log-loss 0.5050 vs 0.5054), and their replays
differ by five points of ROI and disagree on whether 2025 made money. That is how noisy a
replay of a few hundred bets is. The log-loss columns above are the evidence that the model
improved; the ROI column is only evidence that the sign is probably positive.

Varying one rule at a time with v4 (not to pick the best, to see whether the result is fragile):
ROI stayed between +6.7% and +17.2% for `min_edge` 0.015–0.04, `max_disagreement` 0.25–0.35,
`max_spread` 0.03–0.10, a narrower price band, and `market_weight` 0.5–0.75. 14 of 22 weeks were
profitable. Realised profit per contract was at or above what the blend predicted in three of
the four quartiles of predicted edge.

Reasons to stay sceptical:

- The features were chosen while watching the market board. Each was added for a stated reason
  and kept or dropped on one look, but that is still selection on the test set. Expect the live
  number to be lower.
- 2025 contributes 110 bets. Almost all the evidence is nine weeks of 2026.
- Per-stat ROI is noise at these counts (11 to 211 bets). The stats that looked bad for the
  baseline (rec_yds, rr_yds, pass_comp) are not the ones that look bad for v4 (pass_att,
  pass_comp, td). That is what noise looks like; no stat was switched off.
- Depth is unknown historically. The replay assumes $10 fills at the quote.
- The quote is the last hourly candle before kickoff, often the kickoff price itself. The bot
  stops 5 minutes earlier and would not get exactly that price.

## Train/serve mismatches found

- **Weather.** `temp` and `wind` in the schedule file are filled in after the game and blank
  for upcoming ones. The model was trained on values it never sees at scan time. They are now
  excluded (`features.NOT_KNOWN_PREGAME`). Cost in the backtest: about 0.001. A forecast feed
  would let them back in.
- **Weekly expert rank before Friday.** A scan run on Tuesday has no weekly rank yet. The model
  handles that (Thursday games never have one in training), but it is less accurate. Scan from
  Friday evening.
- **Who is "available".** In history that is everyone who played; for an upcoming game it is the
  active roster minus Out and Doubtful. A Questionable player who ends up inactive is counted
  as present. The bot already skips Questionable players.

## Next

1. Paper trade. Nothing replaces prices the bot actually saw at the time it saw them.
2. Weather forecasts for outdoor games.
3. Sportsbook lines (`odds.py`, still untested, needs a key). Books move on news faster than
   anything in this model; the expert ranks are a weekly proxy for the same information.
4. Log every scan's quotes to disk. That builds a price history with real depth and real
   timestamps, which the replay does not have.
5. Re-run `market-backtest --refresh` every few weeks and watch whether the blend gain holds.
