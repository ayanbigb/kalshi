# Combo strategy: the math, the evidence, and what the bot does

Everything below is reproducible:

```bash
python -m propbot fit-realization                 # 1. how much of the model's edge is real
python -m propbot combo-backtest-market           # 2. combos on real 2025-26 prices, combinable stats only
python -m propbot combo-backtest-market --all-stats   #    same, including rec / rec yds / rush yds legs
python -m propbot combo-math --p-leg 0.6 --edge 0.05 --hold 0.10   # 3. 3 legs vs 5 legs, same-quality legs
python -m propbot combo-plan --bankroll 500 --weeks 8 --per-week 3 --target 0.20   # 4. stakes and risk
python -m propbot combo-run                       # 5. the policy: choose, size, record PAPER combos
python -m propbot combo-settle
python -m propbot run --loop 15 --combos          #    singles + combos every 15 minutes
```

## 1. Is the model's disagreement with Kalshi real? (edge realization)

On 48,529 settled, tradeable Kalshi prop markets (2025 wk 6 – 2026 wk 9), fit

    P(yes) = expit( logit(p_market) + λ · (logit(p_blend) − logit(p_market)) )

where `p_blend` is the bot's 70/30 market/model blend. λ = 1 means the blend's disagreement with the market
came true on average; λ = 0 means the market was right.

| | λ | 10%–90% (game bootstrap) |
|---|---|---|
| pooled | **1.13** | 0.91 – 1.34 |
| 2025 | 1.08 | |
| 2026 | 1.14 | |

Per stat (shrunk toward the pooled value; `lo` is what the bot sizes on):

| stat | n | raw | shrunk | lo |
|---|---|---|---|---|
| pass_yds | 3,951 | 1.67 | 1.43 | 1.22 |
| rec | 11,074 | 1.33 | 1.28 | 1.07 |
| pass_att | 405 | 2.65 | 1.31 | 1.09 |
| td | 3,584 | 0.97 | 1.04 | 0.83 |
| rush_yds | 8,764 | 0.85 | 0.92 | 0.71 |
| rec_yds | 16,447 | 0.83 | 0.88 | 0.66 |
| pass_int | 670 | 0.11 | 0.94 | 0.72 |

So the blend's edge is real on average, and the 70/30 weight is if anything too timid. But it is weakest
exactly on the most-traded stats (receiving and rushing yards). Every leg the bot puts in a combo is re-priced as

    p_robust = expit( logit(p_market) + min(λ_lo(stat), 1) · gap )

i.e. it keeps only the part of its edge that the data supports with ~90% confidence, and never leans further
from the market than the blend itself.

## 2. Combos on real prices, by leg count

Each week, legs come from the walk-forward model and the real last pre-kickoff mid; correlations were fitted
only on earlier seasons and λ only on earlier weeks; the bot's optimizer picks the best (log-growth) combos of
each size 2–6 (up to 3 non-overlapping per size per week) and they are settled on what happened. Kalshi does
not publish combo quotes, so each is priced at its correlation-aware fair value × (1 + hold). ROI is per $1
staked at a 10% hold; intervals resample whole weeks.

**Only stats Kalshi lets into combos** (no rec / rec yds / rush yds), 187 combos:

| legs | combos | model said | hit | price | ROI | 90% CI |
|---|---|---|---|---|---|---|
| 2 | 59 | 25.6% | 37.3% | 0.233 | +64% | +7% … +127% |
| 3 | 53 | 19.3% | 20.8% | 0.164 | +100% | −7% … +214% |
| 4 | 48 | 15.4% | 16.7% | 0.124 | +167% | −14% … +354% |
| 5 | 15 | 25.2% | 26.7% | 0.191 | +160% | −63% … +482% |
| 6 | 12 | 27.1% | 25.0% | 0.198 | +309% | −70% … +778% |

**All stats**, 278 combos:

| legs | combos | model said | hit | price | ROI | 90% CI |
|---|---|---|---|---|---|---|
| 2 | 76 | 34.3% | 31.6% | 0.319 | −16% | −52% … +26% |
| 3 | 71 | 27.2% | 28.2% | 0.235 | +16% | −38% … +76% |
| 4 | 58 | 19.6% | 15.5% | 0.159 | −25% | −74% … +38% |
| 5 | 41 | 18.6% | 14.6% | 0.142 | +60% | −79% … +239% |
| 6 | 32 | 19.1% | 9.4% | 0.138 | +5% | −88% … +138% |

What survives both runs (465 combos together):

- **Overall calibration is right**: predicted 23.7%, hit 23.7%.
- **The legs beat the market's own price**: hit rate ÷ market fair price = 1.38 (2 legs), 1.40 (3), 1.29 (4),
  1.34 (5), 1.02 (6). Whether that becomes profit depends on the hold Kalshi's market makers charge.
- **Overconfidence grows with legs**: hit ÷ predicted = 1.12, 1.05, 0.90, 0.88, 0.64 for 2…6 legs. The
  optimizer picks the legs the model is most optimistic about, and that selection compounds per leg. The bot
  multiplies combo probabilities by a factor fitted from this (`models/combo_calibration.json`: 1.00, 1.00,
  0.95, 0.95, 0.86; it only ever shrinks), and does not bet 6-leg combos by default.
- **Noise is enormous**: no row's interval excludes zero in both runs. The combinable-only run ran hot (2-leg
  hit 37% against 26% predicted is about 2 standard errors); do not plan on those ROIs.

## 3. Why the number of legs matters (same-quality legs)

For legs that each beat the market by a factor (1 + e), a k-leg combo priced at the market's fair value
times (1 + h) returns, per $1,

    ROI(k) ≈ (1 + e)^k / (1 + h) − 1          break-even legs  k* = ln(1 + h) / ln(1 + e)

while its standard deviation per $1 grows roughly like 1 / √(p_leg^k). Kelly stake, growth and the number of
bets needed before the edge is visible all follow. `combo-math`, our P 60% per leg:

**+5% edge per leg, 10% hold**

| legs | P(win) | payout | EV/$ | sd/$ | Kelly | ¼-Kelly stake on $500 | bets to be 95% sure |
|---|---|---|---|---|---|---|---|
| 1 | 60.0% | 1.70x | +1.9% | 0.83 | 2.8% | $3.47 | 4,972 |
| 2 | 36.0% | 2.66x | −4.1% | 1.28 | 0 | — | — |
| 3 | 21.6% | 4.62x | −0.3% | 1.90 | 0 | — | — |
| 4 | 13.0% | 8.03x | +4.1% | 2.70 | 0.6% | $0.72 | 11,882 |
| 5 | 7.8% | 14.0x | +8.9% | 3.75 | 0.7% | $0.86 | 4,792 |
| 6 | 4.7% | 24.5x | +14.1% | 5.16 | 0.6% | $0.75 | 3,602 |

**+8% edge per leg, 5% hold**

| legs | P(win) | payout | EV/$ | sd/$ | Kelly | ¼-Kelly stake | bets to be 95% sure |
|---|---|---|---|---|---|---|---|
| 1 | 60.0% | 1.75x | +4.7% | 0.86 | 6.4% | $7.95 | 880 |
| 2 | 36.0% | 2.95x | +6.1% | 1.41 | 3.1% | $3.90 | 1,470 |
| 3 | 21.6% | 5.25x | +13.5% | 2.16 | 3.2% | $3.96 | 698 |
| 4 | 13.0% | 9.41x | +21.9% | 3.16 | 2.6% | $3.26 | 563 |
| 5 | 7.8% | 16.9x | +31.3% | 4.52 | 2.0% | $2.46 | 566 |
| 6 | 4.7% | 30.3x | +41.5% | 6.40 | 1.4% | $1.77 | 642 |

Read it as:

- **The hold decides the minimum leg count.** With a 10% hold and 5% legs, 2–3-leg combos lose money; the edge
  only outruns the hold from 4 legs. With a 5% hold and 8% legs, everything is positive.
- **ROI keeps rising with legs, but the Kelly stake falls** because variance rises faster. Long-run growth
  (what Kelly maximises) peaks around 3–5 legs and then declines.
- **Hundreds of bets** separate a real edge from luck at any leg count. A season of combos cannot prove the
  strategy works; paper trading measures it, it does not validate it.
- Positive correlation between legs (same game, same direction) raises P(win) above the product. It helps only
  if the market maker prices it less than you do; `combos` prices the fair value with the same correlations, so
  correlation alone is not counted as edge.

## 4. How much to stake (combo-plan)

From the real-price backtest, using the lower of the backtest hit rate and the model's prediction, with the
hit rate drawn from a Beta posterior in each simulated season (so being wrong about the edge is part of the
risk). $500 bankroll, 8 weeks × 3 combos, 10% hold, ¼ Kelly. All-stats backtest:

| legs | hit | price | payout | EV/$ | Kelly | ¼-Kelly stake | median 8-wk return | P(profit) | 95th-pct drawdown |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 32.1% | 0.304 | 3.1x | +0.6% | 0.3% | $0.33 | +0.1% | 53% | 1% |
| 3 | 27.2% | 0.223 | 4.3x | +15.7% | 4.8% | $6.04 | +1.3% | 65% | 17% |
| 4 | 16.7% | 0.149 | 6.3x | +5.2% | 1.0% | $1.23 | +0.3% | 57% | 5% |
| 5 | 16.3% | 0.133 | 7.1x | +15.1% | 2.5% | $3.11 | +2.4% | 54% | 13% |
| 6 | 11.8% | 0.130 | 7.3x | −14.5% | 0 | $0 | — | — | — |

And to **target +20% in 8 weeks**, the stake per combo would have to be $26.50 on 3-leg combos (1.1× Kelly;
P(reach target) 49%, P(losing half the bankroll) 17%) or $27.58 on 5-leg ones (2.2× Kelly; P(ruin) 37%).
Over-betting Kelly raises the chance of the target and the chance of ruin together; past 2× Kelly the
expected growth is negative even though the expected profit is not.

## 5. What the bot does (`combo-run`, `run --combos`)

1. Scan Kalshi, price every rung, keep legs that pass the singles gates (spread ≤ 6¢, |model − market| ≤ 0.20,
   ≥ 6 career games, not Questionable) and that Kalshi accepts in combos (exchange collection list minus
   `not_combinable_stats`, never `…PROT`).
2. Optimise combos (beam search on expected log growth) with caps: ≤ 4 legs per game, ≤ 2 per player, ≤ 2 of one
   stat type, no pair with |correlation| > 0.85.
3. Re-price each candidate with `p_robust` legs and the leg-count calibration factor; keep it only if, at the
   expected quote, robust EV ≥ `min_roi` (10%) and its leg count is in `allowed_legs` ([2, 3, 4, 5]).
4. Stake `combo_kelly_fraction` (0.10) × robust Kelly × bankroll, capped at `max_stake_per_combo` ($10) and
   `max_weekly_combo_frac` (5% of bankroll open at once), at most `max_combos_per_run` (3).
5. Record as PAPER combos with the price to ask for and the most to accept; `combo-settle` settles leg by leg.
   Combo orders go through Kalshi's request-for-quote flow, which this code does not automate: place them in
   the app, and only if the quote is at or below "accept up to".

## Caveats

- Combo prices (the hold) are assumed. Everything above moves with it; check the real quote every time.
- The backtest uses the last hourly quote before kickoff and assumes you could trade at it.
- 465 backtest combos over 27 weeks is a small sample for bets that win 10–35% of the time.
- Kalshi's combo eligibility changes; the exchange's collection list is read live, the Help Center exclusions
  are a config list.
