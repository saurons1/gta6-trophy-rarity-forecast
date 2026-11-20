"""GTA VI trophy rarity forecast (make.fun bounty gta6-trophy-rarity-forecast-01).

Forecasts every GTA VI trophy's Exophase earn rate at a case's target time as
five quantiles, from the case's history of Exophase reads.

Model (see README.md for the full reasoning):
  * Each read gives the players tracked N and every trophy's earn rate r.
    A trophy's earners are e = r/100 * N.
  * Two estimators of the target rate are blended in logit space:
    (A) a recency-weighted trend of logit(r) against ln(1 + hours since
    release), its slope shrunk toward the cross-trophy median slope;
    (B) a mechanistic split: power-law fits of ln N and ln e against the
    same log-time, then rate = e*/N* at the target.  The split is what
    anticipates dilution: when the tracked base grows faster than a
    trophy's earners, the rate falls even as more players earn it.
  * The extrapolated move is damped (0.85) toward the last read, because a
    4.5-day log-time extrapolation must not run away from the data.
  * The spread is the larger fit residual plus a random-walk term growing
    with the horizon plus a floor for Exophase's own refresh jumps, which
    move every rate at once (the launch-night refresh took the tracked
    base 638 -> 3,868 and every rate but the 100% one down).  The band is
    deliberately honest: a narrow band that misses a refresh costs far
    more pinball loss than a wide band costs when the answer lands near
    the median.

Standard library only, no network, deterministic: the same case always
writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Tuning constants (see README.md for why each is what it is).
SIGMA_RW = 0.5          # logit spread per sqrt(day) of horizon (random walk)
SIGMA_FLOOR = 0.8       # logit floor: one Exophase refresh can move every rate
SIGMA_RESID_FLOOR = 0.25
DAMP = 0.85             # shrink the extrapolated move toward the last read
SHRINK_K = 2.0          # pseudo-count for slope shrinkage toward the median
HALF_LIFE_H = 24.0      # recency-weight half-life for the trend fits (hours)
PRIOR_B_N = 0.7         # prior power-law exponent for tracked-player growth
PRIOR_B_E = 0.3         # prior power-law exponent for earner growth
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def hours_since_release(iso: str) -> float:
    """Hours from 2026-11-19T00:00:00Z to an ISO-8601 UTC timestamp."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return (dt - RELEASE).total_seconds() / 3600.0


def log_time(t_hours: float) -> float:
    """log-time axis: ln(1 + hours since release), floored so it stays finite."""
    return math.log1p(max(t_hours, -0.9))


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def wls(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares of y on x. Returns (intercept, slope, resid_std)."""
    sw = sum(ws)
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * y for w, y in zip(ws, ys))
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    denom = sw * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (sw * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / sw
    var = sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return intercept, slope, math.sqrt(max(var, 0.0))


def distinct_states(history: list[dict]) -> list[dict]:
    """Collapse consecutive identical reads (Exophase holds the page between
    refreshes, so most reads are duplicates) to one state each, first-seen."""
    states: list[dict] = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            states.append(read)
            prev_key = key
    return states


def shrink(slope: float | None, prior: float, n_eff: float) -> float:
    """Shrink a fitted slope toward a prior; no fit means the prior itself."""
    if slope is None:
        return prior
    return (n_eff * slope + SHRINK_K * prior) / (n_eff + SHRINK_K)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    t_target = hours_since_release(case["target"])
    x_target = log_time(t_target)

    if not history:
        # No history at all: a flat, wide guess beats crashing (an invalid
        # case scores as the last rate and takes no prize).
        return {"trophies": [{"id": t["id"], "q": [0.05, 0.25, 1.0, 4.0, 20.0]} for t in trophies]}

    states = distinct_states(history)
    t_last = hours_since_release(states[-1]["at"])
    x_last = log_time(t_last)
    n_last = float(states[-1]["players_tracked"])
    n_points = [(log_time(hours_since_release(s["at"])), math.log(max(float(s["players_tracked"]), 1.0))) for s in states]

    # --- pass 1: fit every trophy's trend and earner growth, collect priors ---
    # Recency weights: a read's weight halves every HALF_LIFE_H hours.
    def weights(times: list[float]) -> list[float]:
        return [0.5 ** ((t_last - t) / HALF_LIFE_H) for t in times]

    fit_a: dict[int, tuple[float, float, float] | None] = {}
    fit_e: dict[int, tuple[float, float, float] | None] = {}
    slopes_a: list[float] = []
    slopes_e: list[float] = []
    for i, trophy in enumerate(trophies):
        pts = [(log_time(hours_since_release(s["at"])), logit(s["rates"][i]))
               for s in states if s["rates"][i] is not None]
        ts = [hours_since_release(s["at"]) for s in states if s["rates"][i] is not None]
        fa = wls([p[0] for p in pts], [p[1] for p in pts], weights(ts)) if len(pts) >= 2 and len({p[0] for p in pts}) >= 2 else None
        fit_a[i] = fa
        if fa is not None:
            slopes_a.append(fa[1])
        epts = [(log_time(hours_since_release(s["at"])),
                 math.log(max(s["rates"][i] / 100.0 * float(s["players_tracked"]), 0.5)))
                for s in states if s["rates"][i] is not None and s["rates"][i] > 0]
        ets = [hours_since_release(s["at"]) for s in states if s["rates"][i] is not None and s["rates"][i] > 0]
        fe = wls([p[0] for p in epts], [p[1] for p in epts], weights(ets)) if len(epts) >= 2 and len({p[0] for p in epts}) >= 2 else None
        fit_e[i] = fe
        if fe is not None:
            slopes_e.append(fe[1])

    slope_a_prior = median(slopes_a) if slopes_a else 0.0
    slope_e_prior = median(slopes_e) if slopes_e else PRIOR_B_E

    # Tracked-player growth (shared by every trophy's mechanistic split).
    n_times = [hours_since_release(s["at"]) for s in states]
    fit_n = wls([p[0] for p in n_points], [p[1] for p in n_points], weights(n_times)) if len(n_points) >= 2 and len({p[0] for p in n_points}) >= 2 else None
    slope_n = shrink(fit_n[1] if fit_n else None, PRIOR_B_N, float(len(n_points)))
    intercept_n = fit_n[0] if fit_n else math.log(max(n_last, 1.0)) - slope_n * x_last
    n_star = math.exp(intercept_n + slope_n * x_target)
    n_star = min(max(n_star, n_last), 10.0 * n_last)

    # --- pass 2: forecast each trophy ---
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = next((s["rates"][i] for s in reversed(states) if s["rates"][i] is not None), None)
        if last_rate is None:
            rows.append({"id": trophy["id"], "q": [0.05, 0.25, 1.0, 4.0, 20.0]})
            continue
        if last_rate >= 99.99:
            # Everybody tracked has it (the starter trophy); it stays ~100.
            rows.append({"id": trophy["id"], "q": [99.77, 99.86, 99.9, 99.94, 99.98]})
            continue
        if last_rate <= 0.0 and all((s["rates"][i] or 0.0) <= 0.0 for s in states):
            # Nobody has earned it yet in 2.5 days: a small positive rate by
            # day seven, with a wide band.
            rows.append({"id": trophy["id"], "q": [0.005, 0.015, 0.05, 0.17, 0.6]})
            continue

        # Estimator A: logit trend, slope shrunk toward the trophy median.
        fa = fit_a[i]
        slope_a = shrink(fa[1] if fa else None, slope_a_prior, float(len(states)))
        intercept_a = fa[0] if fa else logit(last_rate) - slope_a * x_last
        y_fit = intercept_a + slope_a * x_target
        y_a = logit(last_rate) + DAMP * (y_fit - logit(last_rate))

        # Estimator B: earner count and tracked base grown separately.
        fe = fit_e[i]
        slope_e = shrink(fe[1] if fe else None, slope_e_prior, float(len(states)))
        e_last = max(last_rate / 100.0 * n_last, 0.5)
        if fe is not None:
            e_star = math.exp(fe[0] + slope_e * x_target)
        else:
            # No earner fit: earners grow with the base at the prior rate.
            e_star = e_last * (n_star / n_last) ** (PRIOR_B_E / PRIOR_B_N)
        e_star = min(max(e_star, 0.0), n_star)
        rate_b = clamp(100.0 * e_star / n_star)
        y_b = logit(rate_b)

        # Blend the two estimators in logit space.
        y_med = 0.5 * (y_a + y_b)

        # Spread: the larger fit residual, a random-walk term that grows
        # with the horizon, and a floor for Exophase refresh jumps.
        resid = SIGMA_RESID_FLOOR
        if fa is not None:
            resid = max(resid, fa[2])
        if fe is not None:
            resid = max(resid, fe[2])
        sigma = math.sqrt(resid ** 2 + (SIGMA_RW ** 2) * horizon_days + SIGMA_FLOOR ** 2)

        q = [round4(clamp(sigmoid(y_med + z * sigma))) for z in Z]
        # Rounding can only keep order; enforce it explicitly anyway.
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
