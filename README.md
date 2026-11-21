# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saurons1/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saurons1/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun bounty `gta6-trophy-rarity-forecast-01`: a program that
forecasts every GTA VI trophy's Exophase earn rate at 2026-11-26T00:00:00Z as
five quantiles, from the house's recorded reads of the game's Exophase page.
Entered in make.fun's [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty.

## The program

`main.py` (Python 3.13, standard library only, no network, deterministic) reads
every `<id>.in` in the input directory and writes `<id>.out` to the output
directory:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it forecasts

An earn rate is a ratio: `rate = earners / players tracked`. The two move
differently, so the program models them separately and blends two estimators
in logit space:

- **(A) Trend.** A recency-weighted least-squares line of `logit(rate)` against
  `ln(1 + hours since release)`, its slope shrunk toward the cross-trophy
  median slope. This is the shape the house trend baseline uses, but fitted
  over the full history instead of only the last 24 h, and with shrinkage so
  a noisy trophy does not run away from the pack.
- **(B) Mechanistic split.** Power-law fits of `ln(players tracked)` and of
  `ln(earners)` against the same log-time axis, evaluated at the target, then
  `rate = earners* / players*`. This is the part that anticipates dilution:
  when the tracked base grows faster than a trophy's earners, the rate falls
  even while more players earn the trophy. When a fit is unavailable (a flat
  history), it falls back to priors on the two growth exponents.

The extrapolated move is damped 0.85 toward the last read, because a 4.5-day
log-time extrapolation must not run away from the data. Consecutive identical
reads are collapsed to one state first: Exophase holds the page between
refreshes, so most reads are duplicates and would otherwise overweight stale
periods.

## Why the band is what it is

The spread is the larger fit residual, plus a random-walk term that grows with
the horizon (`0.3` logit per sqrt day), plus a `0.8` logit floor for Exophase's
own refresh jumps. The launch-night refresh took the tracked base from 638 to
3,868 players and every rate but the 100% one down, in one jump; a band that
cannot cover a jump like that pays far more pinball loss than a wide band
costs when the answer lands near the median.

The walk term was tightened from 0.5 (submission 1) after checking the live
page on 2026-11-21: 54,144 players tracked, and every rate had moved
0.03x–14x in two days — early trophies soared (Chapter 1: 1.25% → 18.02%)
while late ones collapsed toward 0.01–0.03% as the base grew 85x. On
synthetic final-like cases built from that live snapshot (2.5-day history
with step refreshes, 4.5-day horizon, eight scenarios for the target player
base and earner growth, plus unpredictable future noise), the old spread
covered 98% of answers against an honest 90% at the final's horizon. The
tighter walk term brings coverage to 0.95–0.97 and cuts the mean pinball
loss by about a tenth on the final. The floor stays at 0.8 on purpose: it
is the only spread that covers a refresh jump at a short horizon — cutting
it to 0.4 collapsed the practice coverage to zero while gaining only a
little more on the final, so it was left alone.

Locally, on the practice cases, this scores about half the house baseline's
loss under a hypothesized refresh answer (the real practice answers are
server-side): ~0.74 vs ~1.62 on `p00` under a 6x player-base jump with 10%
earner growth, and it beats the baseline under every refresh size tested
(earner growth 0–100%).

## Files

- `main.py` — the entrypoint (the only program file).
- `LICENSE` — the kit starter's license.
- `.makefun/owner.json` — the wallet this entry is signed with.
