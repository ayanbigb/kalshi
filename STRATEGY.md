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
| pooled | **1.19** | 0.96 – 1.43 |
| 2025 | 1.05 | |
| 2026 | 1.23 | |

Per stat (shrunk toward the pooled value; `lo` is what the bot sizes on, capped at 1):

| stat | n | raw | shrunk | lo |
|---|---|---|---|---|
| pass_yds | 3,951 | 1.96 | 1.62 | 1.40 |
| pass_comp | 437 | 2.31 | 1.33 | 1.10 |
| pass_att | 405 | 2.23 | 1.31 | 1.08 |
| rush_att | 1,019 | 1.42 | 1.25 | 1.02 |
| pass_int | 670 | 1.49 | 1.24 | 1.01 |
| td | 3,584 | 1.11 | 1.14 | 0.92 |
| rec_yds | 16,447 | 1.12 | 1.13 | 0.90 |
| pass_tds | 1,519 | 1.02 | 1.13 | 0.90 |
| rec | 11,074 | 1.07 | 1.09 | 0.87 |
| rr_yds | 659 | 0.42 | 1.05 | 0.82 |
| rush_yds | 8,764 | 0.89 | 0.97 | 0.74 |

So the blend's edge is real on average, and the 70/30 weight is if anything too timid. It is weakest on
rushing yards and rush+rec yards. Every leg the bot puts in a combo is re-priced as

    p_robust = expit( logit(p_market) + min(λ_lo(stat), 1) · gap )

i.e. it keeps only the part of its edge that the data supports with ~90% confidence, and never leans further
from the market than the blend itself.

## 2. Combos on real prices, by leg count

Each week, legs come from the walk-forward model and the real last pre-kickoff mid; correlations were fitted
only on earlier seasons and λ only on earlier weeks; the bot's optimizer picks the best (log-growth) combos of
each size 2–6 (up to 3 non-overlapping per size per week) and they are settled on what happened. Kalshi does
not publish combo quotes, so each is priced at its correlation-aware fair value × (1 + hold). ROI is per $1
staked at a 10% hold; intervals resample whole weeks.

**Only stats Kalshi lets into combos** (no rec / rec yds / rush yds), 186 combos:

| legs | combos | model said | hit | price | ROI | 90% CI |
|---|---|---|---|---|---|---|
| 2 | 57 | 23.5% | 28.1% | 0.213 | +62% | −5% … +133% |
| 3 | 53 | 20.2% | 20.8% | 0.173 | −6% | −73% … +75% |
| 4 | 45 | 15.5% | 17.8% | 0.126 | +36% | −74% … +160% |
| 5 | 17 | 20.9% | 29.4% | 0.158 | +174% | −61% … +509% |
| 6 | 14 | 21.3% | 35.7% | 0.153 | +324% | −41% … +860% |

**All stats**, 284 combos:

| legs | combos | model said | hit | price | ROI | 90% CI |
|---|---|---|---|---|---|---|
| 2 | 76 | 33.3% | 27.6% | 0.311 | −26% | −58% … +14% |
| 3 | 71 | 26.7% | 18.3% | 0.234 | −49% | −70% … −26% |
| 4 | 63 | 19.0% | 15.9% | 0.155 | −3% | −67% … +93% |
| 5 | 43 | 17.8% | 16.3% | 0.136 | −41% | −73% … −8% |
| 6 | 31 | 17.7% | 9.7% | 0.129 | −53% | −88% … −9% |

**Both together**, 470 combos (ROI per $1 at 0% / 5% / 10% hold):

| legs | combos | model said | hit | market fair | hit ÷ fair | ROI @0% | ROI @5% | ROI @10% [90% CI] |
|---|---|---|---|---|---|---|---|---|
| 2 | 133 | 29.1% | 27.8% | 23.5% | 1.18 | +23% | +17% | +12% [−28%, +55%] |
| 3 | 124 | 23.9% | 19.4% | 18.1% | 1.07 | −24% | −27% | −31% [−63%, +3%] |
| 4 | 108 | 17.6% | 16.7% | 12.4% | 1.35 | +24% | +19% | +13% [−48%, +84%] |
| 5 | 60 | 18.7% | 20.0% | 12.3% | 1.63 | +32% | +26% | +20% [−59%, +129%] |
| 6 | 45 | 18.8% | 17.8% | 11.8% | 1.51 | +81% | +72% | +64% [−67%, +253%] |

**The robustness check that matters most.** The same two backtests run with the previous model (log-loss
0.5046 vs 0.5033 on single props: nearly the same model) gave, for all stats at a 10% hold, 2 legs −16%,
3 legs **+16%**, 4 legs −25%, 5 legs **+60%**, 6 legs +5%. A near-identical model flips the sign of several
rows. Per-leg-count combo P&L over a season and a half is noise.

What does survive every run:

- **Combos are slightly overconfident**: 22.8% predicted, 21.1% hit over 470 combos (the optimizer picks the
  legs the model likes most, and that selection compounds per leg). The bot multiplies combo probabilities by a
  shrink-only factor fitted from this (`models/combo_calibration.json`: 0.97, 0.87, 0.97, 1.00, 0.98 for 2–6
  legs).
- **The legs still beat the market's own price**: hit ÷ market fair = 1.07 – 1.63 in every row. The edge is
  real before the market maker's hold; whether it survives the hold is the open question, and with a 10% hold
  the answer in this sample is "not reliably, at any leg count".
- **No leg count is proven better.** The analytic section below is the better guide to 3 vs 5 legs.

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
risk). $500 bankroll, 8 weeks × 3 combos, 10% hold, ¼ Kelly.

Combinable stats only:

| legs | hit | price | payout | EV/$ | Kelly | median 8-wk return | P(profit) | stake for +20% | × Kelly | P(reach) | P(lose half) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 23.5% | 0.201 | 4.7x | +10.5% | 2.8% | +2.8% | 50% | $39.72 | 2.8 | 50% | 39% |
| 3 | 20.2% | 0.163 | 5.8x | +17.4% | 3.6% | +4.1% | 54% | $23.97 | 1.3 | 54% | 22% |
| 4 | 15.5% | 0.118 | 8.0x | +23.9% | 3.4% | +2.8% | 50% | $17.40 | 1.0 | 50% | 18% |
| 5 | 20.9% | 0.148 | 6.4x | +33.2% | 6.2% | +10.8% | 67% | $12.56 | 0.4 | 39% | 4% |
| 6 | 21.3% | 0.144 | 6.6x | +39.8% | 7.2% | +14.0% | 67% | $10.46 | 0.3 | 41% | 3% |

All stats (closer to the whole market; 2, 3 and 6 legs have no edge at a 10% hold, so Kelly stakes nothing):

| legs | hit | price | EV/$ | Kelly | median 8-wk return | P(profit) | stake for +20% | × Kelly | P(lose half) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 28.2% | 0.296 | −9.3% | 0 | — | — | — | — | — |
| 3 | 19.2% | 0.221 | −17.9% | 0 | — | — | — | — | — |
| 4 | 16.9% | 0.146 | +9.3% | 1.7% | +0.7% | 58% | $44.95 | 5.3 | 52% |
| 5 | 17.8% | 0.128 | +30.8% | 4.8% | +5.5% | 60% | $13.54 | 0.6 | 4% |
| 6 | 12.1% | 0.121 | −5.8% | 0 | — | — | — | — | — |

The 5- and 6-leg rows of the first table rest on 14–17 combos and look best only because those few hit; the
second table, with 31–43 of them, disagrees. Use these as orders of magnitude: at a sane fraction of Kelly,
combos are a few dollars per bet on a $500 bankroll, and a +20% season needs stakes at or above full Kelly,
where the chance of losing half the bankroll is 20–50%. Over-betting Kelly raises the chance of the target
and the chance of ruin together; past 2× Kelly expected growth is negative even when expected profit is not.

## 5. What the bot does (`combo-run`, `run --combos`)

1. Scan Kalshi, price every rung, keep legs that pass the singles gates (spread ≤ 6¢, |model − market| ≤ 0.20,
   ≥ 6 career games, not Questionable) and that Kalshi accepts in combos (exchange collection list minus
   `not_combinable_stats`, never `…PROT`).
2. Optimise combos (beam search on expected log growth) with caps: ≤ 4 legs per game, ≤ 2 per player, ≤ 2 of one
   stat type, no pair with |correlation| > 0.85.
3. Re-price each candidate with `p_robust` legs and the leg-count calibration factor; keep it only if, at the
   expected quote, robust EV ≥ `min_roi` (10%), robust P(win) ≥ `policy_min_p` (10%: the backtest only has
   evidence for combos that win 10–35% of the time, and long shots are where per-leg error compounds most) and
   its leg count is in `allowed_legs` ([2, 3, 4, 5]). Since no leg count is proven better (section 2), the
   policy lets robust EV and growth choose the count each week rather than fixing one.
4. Stake `combo_kelly_fraction` (0.10) × robust Kelly × bankroll, capped at `max_stake_per_combo` ($10) and
   `max_weekly_combo_frac` (5% of bankroll open at once), at most `max_combos_per_run` (3).
5. Record as PAPER combos with the price to ask for and the most to accept; `combo-settle` settles leg by leg.
   Combo orders go through Kalshi's request-for-quote flow, which this code does not automate: place them in
   the app, and only if the quote is at or below "accept up to".

## Caveats

- Combo prices (the hold) are assumed. Everything above moves with it; check the real quote every time.
- The backtest uses the last hourly quote before kickoff and assumes you could trade at it.
- 465 backtest combos over 27 weeks is a small sample for bets that win 10–35% of the time.
- Kalshi's combo eligibility changes; the exchange's collection list is read live, extra exclusions
  are a config list.
