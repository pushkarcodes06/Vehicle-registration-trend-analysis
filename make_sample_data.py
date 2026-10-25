#!/usr/bin/env python3
"""Generate a SYNTHETIC monthly vehicle-registration dataset.

This is NOT real registration data. It exists so the analysis pipeline can be run and tested
end to end without downloading anything. Every number below is an assumption chosen to look
plausible (a growth trend, festive seasonality, a lockdown collapse, an EV S-curve). Findings
computed from it describe these assumptions, not the real market.

Output columns (the same schema the pipeline expects from a real export):
    date            first day of the month (YYYY-MM-DD)
    state           state name
    vehicle_class   Two-Wheeler | Passenger Car | Three-Wheeler | Goods Vehicle | Bus
    fuel            Petrol | Diesel | CNG | Electric | Hybrid
    registrations   count of new registrations in that month
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Two-Wheeler", "Passenger Car", "Three-Wheeler", "Goods Vehicle", "Bus"]
FUELS = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]

# state: (volume scale, growth adjustment per year, EV timing shift in months (negative = earlier),
#         passenger-car tilt, diesel factor, CNG factor)
STATES = {
    "Maharashtra": (1.55, 0.000, -4, 1.10, 1.00, 1.00),
    "Uttar Pradesh": (1.50, 0.010, 8, 0.80, 1.10, 1.20),
    "Tamil Nadu": (1.20, 0.000, -2, 0.95, 1.00, 0.70),
    "Karnataka": (1.00, 0.015, -8, 1.10, 0.90, 0.60),
    "Gujarat": (0.95, 0.005, 0, 1.00, 0.90, 1.80),
    "Rajasthan": (0.80, 0.010, 6, 0.85, 1.20, 0.80),
    "Kerala": (0.70, -0.005, -3, 1.15, 0.80, 0.50),
    "Delhi": (0.45, -0.020, -14, 1.35, 0.30, 2.20),
    "West Bengal": (0.60, 0.000, 4, 0.85, 1.00, 0.70),
    "Telangana": (0.65, 0.020, -5, 1.05, 0.90, 0.60),
}
CLASS_BASE = {"Two-Wheeler": 72_000, "Passenger Car": 14_000, "Three-Wheeler": 3_200, "Goods Vehicle": 5_000, "Bus": 600}
CLASS_GROWTH_ADJ = {"Two-Wheeler": -0.005, "Passenger Car": 0.030, "Three-Wheeler": 0.010, "Goods Vehicle": 0.015, "Bus": 0.0}
CLASS_SEASON_AMP = {"Two-Wheeler": 1.0, "Passenger Car": 0.9, "Three-Wheeler": 0.6, "Goods Vehicle": 0.8, "Bus": 0.5}
CLASS_SHOCK_SEVERITY = {"Two-Wheeler": 1.0, "Passenger Car": 1.0, "Three-Wheeler": 1.1, "Goods Vehicle": 0.85, "Bus": 1.15}

# Underlying demand growth per calendar year (before shocks)
ANNUAL_GROWTH = {2019: -0.04, 2020: 0.01, 2021: 0.08, 2022: 0.10, 2023: 0.08, 2024: 0.045, 2025: 0.035, 2026: 0.03}
# Monthly seasonal shape, Jan..Dec: fiscal-year-end push in March, festive peak in Oct-Nov
SEASON = np.array([1.00, 0.97, 1.14, 0.93, 0.91, 0.88, 0.93, 0.99, 1.03, 1.20, 1.08, 0.94])
SEASON = SEASON / SEASON.mean()
# Lockdown demand multiplier (1.0 = normal), by month
LOCKDOWN = {
    "2020-03": 0.55, "2020-04": 0.04, "2020-05": 0.20, "2020-06": 0.55, "2020-07": 0.72, "2020-08": 0.82,
    "2020-09": 0.90, "2020-10": 1.00, "2021-03": 0.95, "2021-04": 0.78, "2021-05": 0.45, "2021-06": 0.62,
    "2021-07": 0.85, "2021-08": 0.95,
}
# One-off events: (months, classes, multiplier)
EVENTS = [
    {"name": "chip shortage", "months": ["2021-12", "2022-01", "2022-02", "2022-03"], "classes": ["Passenger Car"], "mult": 0.88},
    {"name": "emission-norm pre-buy", "months": ["2023-03"], "classes": ["Two-Wheeler", "Passenger Car"], "mult": 1.08},
    {"name": "emission-norm hangover", "months": ["2023-04"], "classes": ["Two-Wheeler", "Passenger Car"], "mult": 0.94},
]
# EV share = cap / (1 + exp(-k (t - t0))), t in months since the first month; t0 is shifted per state
EV_PARAMS = {
    "Two-Wheeler": {"cap": 0.30, "t0": 110, "k": 0.040},
    "Three-Wheeler": {"cap": 0.70, "t0": 62, "k": 0.050},
    "Passenger Car": {"cap": 0.30, "t0": 140, "k": 0.035},
    "Goods Vehicle": {"cap": 0.15, "t0": 120, "k": 0.040},
    "Bus": {"cap": 0.35, "t0": 100, "k": 0.040},
}
NOISE_SIGMA = 0.035


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def non_ev_mix(cls: str, s: np.ndarray, diesel_f: float, cng_f: float) -> np.ndarray:
    """Fuel mix of non-electric registrations, columns = [Petrol, Diesel, CNG, Hybrid]; s = progress 0..1."""
    zeros = np.zeros_like(s)
    if cls == "Two-Wheeler":
        cols = [1.0 + zeros, zeros, zeros, zeros]
    elif cls == "Passenger Car":
        diesel = (0.38 - 0.21 * s) * diesel_f
        cng = (0.04 + 0.10 * s) * cng_f
        hybrid = 0.005 + 0.045 * s**1.5
        petrol = np.clip(1.0 - diesel - cng - hybrid, 0.05, None)
        cols = [petrol, diesel, cng, hybrid]
    elif cls == "Three-Wheeler":
        cols = [0.22 + zeros, (0.36 - 0.10 * s) * diesel_f, (0.42 + 0.10 * s) * cng_f, zeros]
    elif cls == "Goods Vehicle":
        cols = [0.08 + zeros, (0.85 - 0.10 * s) * diesel_f, (0.07 + 0.10 * s) * cng_f, zeros]
    else:  # Bus
        cols = [zeros, (0.86 - 0.12 * s) * diesel_f, (0.14 + 0.12 * s) * cng_f, zeros]
    mix = np.vstack(cols).T
    return mix / mix.sum(axis=1, keepdims=True)


def generate(start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, end, freq="MS")
    n = len(months)
    t = np.arange(n)
    month_idx = months.month.to_numpy() - 1
    s_prog = t / max(n - 1, 1)
    keys = months.strftime("%Y-%m")
    rows = []
    for state, (scale, g_adj, ev_shift, car_tilt, diesel_f, cng_f) in STATES.items():
        for cls in CLASSES:
            # underlying growth index
            annual = np.array([ANNUAL_GROWTH[y] for y in months.year]) + CLASS_GROWTH_ADJ[cls] + g_adj
            level = np.exp(np.cumsum(annual / 12.0))
            level = level / level[0]
            season = 1.0 + CLASS_SEASON_AMP[cls] * (SEASON[month_idx] - 1.0)
            shock = np.array([LOCKDOWN.get(k, 1.0) for k in keys])
            shock = np.clip(1.0 - CLASS_SHOCK_SEVERITY[cls] * (1.0 - shock), 0.01, 1.0)
            event = np.ones(n)
            for ev in EVENTS:
                if cls in ev["classes"]:
                    event[np.isin(keys, ev["months"])] *= ev["mult"]
            tilt = car_tilt if cls == "Passenger Car" else (1.0 / np.sqrt(car_tilt) if cls == "Two-Wheeler" else 1.0)
            noise = np.exp(rng.normal(-0.5 * NOISE_SIGMA**2, NOISE_SIGMA, n))
            total = CLASS_BASE[cls] * scale * tilt * level * season * shock * event * noise
            total = np.rint(total).astype(np.int64)

            p = EV_PARAMS[cls]
            ev_share = p["cap"] * sigmoid(p["k"] * (t - (p["t0"] + ev_shift)))
            mix = non_ev_mix(cls, s_prog, diesel_f, cng_f)  # Petrol, Diesel, CNG, Hybrid
            shares = np.column_stack([mix[:, 0], mix[:, 1], mix[:, 2], np.zeros(n), mix[:, 3]]) * (1 - ev_share)[:, None]
            shares[:, 3] = ev_share  # Electric, in FUELS order: Petrol, Diesel, CNG, Electric, Hybrid
            shares = shares / shares.sum(axis=1, keepdims=True)
            counts = np.vstack([rng.multinomial(total[i], shares[i]) for i in range(n)])
            for j, fuel in enumerate(FUELS):
                rows.append(pd.DataFrame({"date": months, "state": state, "vehicle_class": cls, "fuel": fuel, "registrations": counts[:, j]}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/registrations_synthetic.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.start, a.end, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    total = df.groupby("date")["registrations"].sum()
    print(f"wrote {out}  rows={len(df):,}  months={df['date'].nunique()}  "
          f"monthly total min/median/max = {total.min():,} / {int(total.median()):,} / {total.max():,}")


if __name__ == "__main__":
    main()
