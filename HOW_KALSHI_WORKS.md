# How Kalshi player props and combos work (with this week's real prices)

Prices below are Kalshi's quotes on 2026-11-14 for Sunday's games (SEA @ LV, KC @ ATL). Rerun any of it:

```bash
python -m propbot ladder "Smith-Njigba" "rec yds"
python -m propbot combo-price "Smith-Njigba rec yds 80+" "Lock pass yds 250+" "Mahomes pass td 2+"
```

## 1. One rung = one $1 contract

"Jaxon Smith-Njigba: 80+ receiving yards" is a contract that pays **$1 if he gets 80 or more**, $0 if not.

- **Price = probability.** If YES trades at 59¢, the market thinks there is about a 59% chance.
- **YES** at the ask (62¢ here) costs 62¢ plus a small fee and returns $1 if it hits: +38¢ profit, a **1.57×** payout.
- **NO** is the other side ("he gets 79 or fewer"). It costs about 1 − YES bid (1 − 0.57 = 43¢) and pays $1 if he misses.
- **Fee** (taker): `ceil(0.07 × contracts × price × (1 − price))`, largest at 50¢ (about 1.75¢ per contract), tiny near 0 or 1.
  Combos use a fee schedule that also charges the market maker.
- Settles on official stats, overtime included. If he is active but never plays a snap, it settles at the
  pre-game fair price instead of YES/NO. "Protected" (…PROT) series settle differently and are not used here.

## 2. The ladder: 80+ vs 100+

Kalshi lists many rungs for the same player. They are the same random number (his yards) cut at different points:

| rung | market | model | lands in this band (market) | payout if YES hits | EV per $1, YES |
|---|---|---|---|---|---|
| 60+ | 76.0% | 72.0% | 8.0% (60–69) | 1.26× | −9% |
| 70+ | 68.0% | 64.7% | 8.5% | 1.38× | −11% |
| **80+** | **59.5%** | 54.9% | 10.5% (80–89) | **1.57×** | −14% |
| 90+ | 49.0% | 45.5% | 6.5% | 1.93× | −12% |
| **100+** | **42.5%** | 39.1% | 11.0% (100–109) | **2.24×** | −13% |
| 120+ | 24.5% | 22.9% | 5.0% | 3.52× | −19% |
| 150+ | 10.5% | 9.8% | 3.0% | 7.25× | −29% |

(model projection: 82.9 yards)

What the ladder means:

- **It always falls.** 100+ can never be likelier than 80+, because every 100-yard game is also an 80-yard game.
- **The gap between rungs is a band.** P(80+) − P(100+) = 59.5% − 42.5% = **17%** is the market's chance he finishes 80–99.
  If you think he will get 85–95, the bet is "80+ YES" plus "100+ NO", not either rung alone.
- **Higher rung → lower probability, bigger payout.** 80+ pays 1.57×, 100+ pays 2.24×, 150+ pays 7.25×.
- **Probability is not value.** What matters is your probability vs the price: EV per $1 = q ÷ (price + fee) − 1.
  This week the model is a few points under the market on every JSN rung, so every JSN YES is negative EV and most
  of the NOs (50+ through 110+) are slightly positive (+1% to +3%). Your "80+ at 75%, 100+ at 45%" example works the same way: 75¢ for 80+ only makes sense
  if you think it is more than ~76% (75¢ + fee); 45¢ for 100+ needs more than ~46%.
- **The best rung is often not the likeliest.** Mahomes passing TDs: 2+ is 53.5% (model 54.5%, EV −2%); **4+ is
  7.5% at a 9¢ ask, model 10.7%: pays 10.4× and is +9.5% EV.** It is the only positive rung on his ladder, and it
  loses 9 times out of 10. The bot picks rungs by EV and Kelly growth, never by hit rate.

## 3. Combos: every leg must hit

A combo pays $1 only if **all** legs hit, so its price is the market maker's estimate of the probability that
all of them happen, plus their margin. Kalshi does not show combo prices up front; you build it and get a quote.

### Different games: multiply

JSN 80+ (SEA @ LV) and Mahomes 2+ pass TDs (KC @ ATL) have nothing to do with each other:
59.5% × 53.5% = **31.8%** (model with correlation: 31.7%). A fair price is ~32¢, about a 3.1× payout.

### Same game: do NOT multiply

Inside one game the legs move together, and multiplying is wrong, sometimes badly:

| combo | multiply the market %s | with correlation | fair payout |
|---|---|---|---|
| JSN 80+ & Drew Lock (his QB) 250+ pass yds | 13.7% | **18.1%** | 5.5× |
| Mahomes 2+ pass TD & Kelce 60+ rec yds | 19.8% | **21.8%** | 4.6× |
| Mahomes 300+ pass yds & Worthy 60+ & Kelce 60+ | 1.0% | **2.7%** | 37× |

Why: if Mahomes throws for 300, someone caught those yards. Given Mahomes 300+, Kelce 60+ goes from 29% to
**49%** and Worthy 60+ from 18% to **36%**. Given JSN 80+, Lock 250+ goes from 20% to **28%**. The higher the
rung, the stronger the tie: given JSN **100+**, Lock 250+ is **32%**.

Correlation also works against you:
- **Teammates competing:** two receivers on one team split targets. "WR1 big game" and "WR2 big game" are less
  likely together than the product says.
- **Game script:** a blowout means the losing QB throws more and the winning RB runs more. "Losing team's RB
  100+ rush yds" plus "losing team's QB 300+ pass yds" fight each other.
- **Same player:** "JSN 6+ catches" and "JSN 80+ yards" are about 80% correlated, nearly one bet counted twice.

### How the bot accounts for it

Each (player, stat) in a game gets a correlated random variable, with correlations measured from past games:
QB yards vs his WR's yards +0.39, QB TDs vs TE yards +0.15, and so on for every role pair, same team or
opposing. It simulates the game 40,000 times and counts how often every leg hits. All rungs of one player share
one variable, so 80+ and 100+ for JSN are consistent with each other and with Lock's passing.

### Is the combo worth it?

Two different questions:

1. **Is the joint probability higher than you would think?** For same-team stacks, yes: correlation is real.
2. **Is it higher than the price?** The market maker knows about correlation too, so the bot assumes their
   quote is the correlated fair price + a margin. Measured on 22 real combos from this account: about +5% for
   5–9 legs, about +27% for 10–13, and +72% to +2,500% for 14+ (the bot uses 6% / 27% / 75%). Correlation alone is not edge; **edge comes only
   from legs the model thinks are underpriced**, and in a combo those edges multiply, but so does any shortfall.

The three same-team stacks above are all overs on stars, and the model is below the market on each leg, so they
come out −28% to −33% EV even though correlation makes them likelier than the product. `combo-price` tells you
the most you should accept for a quote (e.g. 13.4¢ for JSN 80+ & Lock 250+, against an expected quote near 20¢).

## 4. How many legs

Per $1, a combo returns about (1 + edge per leg)^legs ÷ (1 + margin) − 1:

- With a real +5% edge per leg and a 10% margin, 2–3 legs lose money and 4+ legs win on average.
- More legs → bigger payout and bigger EV, but much lower hit rate and wilder swings, so the right stake shrinks.
  Long-run growth peaks around 3–5 legs.
- With no edge per leg, every extra leg just adds margin. More legs only helps if each leg is +EV on its own.

See `STRATEGY.md` for the full math, the backtest on real prices, and the staking plan.
