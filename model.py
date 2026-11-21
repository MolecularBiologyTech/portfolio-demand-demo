"""Risk-adjusted manufacturing demand for a portfolio of client molecules.

Pure functions, no UI: the Streamlit app and the tests both import from here.
"""
import numpy as np
import pandas as pd

def _quiet_blas():
    # macOS Accelerate BLAS raises spurious FP warnings in matmul (results identical to einsum); harmless on Linux.
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


REQUIRED = ["molecule", "phase", "launch_year", "peak_batches", "ramp_years"]


def validate(portfolio: pd.DataFrame, pos_by_phase: dict) -> pd.DataFrame:
    """Reject input the model cannot use, with a message a user can act on."""
    missing = [c for c in REQUIRED if c not in portfolio.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    df = portfolio.dropna(subset=REQUIRED).copy()
    unknown = sorted(set(df["phase"]) - set(pos_by_phase))
    if unknown:
        raise ValueError(f"unknown phase(s): {', '.join(unknown)}")
    if (df["peak_batches"] < 0).any() or (df["ramp_years"] < 1).any():
        raise ValueError("peak_batches must be >= 0 and ramp_years >= 1")
    return df


def demand_curve(launch_year: int, peak: float, ramp_years: int, years: np.ndarray) -> np.ndarray:
    """Batches per year if the molecule succeeds: linear ramp from launch to peak, then flat."""
    t = years - launch_year + 1
    return peak * np.clip(t / ramp_years, 0.0, 1.0)


def success_probability(phase: pd.Series, pos_by_phase: dict, multiplier: float) -> np.ndarray:
    return np.clip(phase.map(pos_by_phase).to_numpy(float) * multiplier, 0.0, 1.0)


def curves(df: pd.DataFrame, years: np.ndarray, delay: int, peak_uplift: float) -> np.ndarray:
    """Matrix molecules x years of demand if every molecule succeeds, after scenario shifts.

    The launch delay applies only to molecules not yet approved: a product on the market cannot launch later.
    """
    return np.vstack([
        demand_curve(r.launch_year + (0 if r.phase == "Approved" else delay),
                     r.peak_batches * peak_uplift, r.ramp_years, years)
        for r in df.itertuples()
    ]) if len(df) else np.zeros((0, len(years)))


def expected_demand(df, pos_by_phase, years, pos_mult=1.0, delay=0, peak_uplift=1.0) -> np.ndarray:
    """Analytic expectation: sum over molecules of P(success) x demand curve."""
    p = success_probability(df["phase"], pos_by_phase, pos_mult)
    with _quiet_blas():
        return p @ curves(df, years, delay, peak_uplift)


def simulate(df, pos_by_phase, years, pos_mult=1.0, delay=0, peak_uplift=1.0,
             n_sims=5000, seed=0) -> np.ndarray:
    """Monte Carlo: each simulation draws which molecules succeed. Returns sims x years."""
    rng = np.random.default_rng(seed)
    p = success_probability(df["phase"], pos_by_phase, pos_mult)
    success = rng.random((n_sims, len(p))) < p
    with _quiet_blas():
        return success.astype(float) @ curves(df, years, delay, peak_uplift)


def summarise(sims: np.ndarray, years: np.ndarray, capacity: float) -> pd.DataFrame:
    """P10/P50/P90 per year and the probability that demand exceeds capacity."""
    return pd.DataFrame({
        "year": years,
        "P10": np.percentile(sims, 10, axis=0),
        "P50": np.percentile(sims, 50, axis=0),
        "P90": np.percentile(sims, 90, axis=0),
        "mean": sims.mean(axis=0),
        "P(demand > capacity)": (sims > capacity).mean(axis=0),
    })
