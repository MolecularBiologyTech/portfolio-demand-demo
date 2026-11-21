import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402

POS = {"Phase 2": 0.2, "Approved": 1.0}
YEARS = np.arange(2025, 2036)


def portfolio(**overrides):
    rows = [dict(molecule="A", phase="Approved", launch_year=2025, peak_batches=10, ramp_years=1),
            dict(molecule="B", phase="Phase 2", launch_year=2028, peak_batches=30, ramp_years=3)]
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


def test_ramp_reaches_peak_and_stays():
    c = model.demand_curve(2028, 30, 3, YEARS)
    assert c[YEARS < 2028].sum() == 0
    assert c[YEARS == 2028][0] == pytest.approx(10)
    assert (c[YEARS >= 2030] == 30).all()


def test_expected_demand_is_probability_weighted_sum():
    df = model.validate(portfolio(), POS)
    exp = model.expected_demand(df, POS, YEARS)
    by_hand = 1.0 * model.demand_curve(2025, 10, 1, YEARS) + 0.2 * model.demand_curve(2028, 30, 3, YEARS)
    np.testing.assert_allclose(exp, by_hand)


def test_monte_carlo_mean_matches_expectation():
    df = model.validate(portfolio(), POS)
    sims = model.simulate(df, POS, YEARS, n_sims=40000, seed=1)
    np.testing.assert_allclose(sims.mean(axis=0), model.expected_demand(df, POS, YEARS), atol=0.5)


def test_delay_shifts_pipeline_but_not_marketed_products():
    df = model.validate(portfolio(), POS)
    approved = model.demand_curve(2025, 10, 1, YEARS)
    base = model.expected_demand(df, POS, YEARS) - approved
    late = model.expected_demand(df, POS, YEARS, delay=2) - approved  # approved molecule unchanged
    np.testing.assert_allclose(late[2:], base[:-2])
    assert late[:2].sum() == 0


def test_multiplier_is_capped_at_certainty():
    df = model.validate(portfolio(), POS)
    p = model.success_probability(df["phase"], POS, 10.0)
    assert p.max() == 1.0


def test_shortfall_probability_bounds():
    df = model.validate(portfolio(), POS)
    sims = model.simulate(df, POS, YEARS, n_sims=2000)
    s = model.summarise(sims, YEARS, capacity=0)
    assert s["P(demand > capacity)"].iloc[-1] == 1.0  # the approved molecule always needs batches
    assert model.summarise(sims, YEARS, capacity=1e9)["P(demand > capacity)"].max() == 0.0


@pytest.mark.parametrize("bad, msg", [
    (dict(phase="Phase 9"), "unknown phase"),
    (dict(peak_batches=-1), "peak_batches"),
    (dict(ramp_years=0), "ramp_years"),
])
def test_validate_rejects_bad_input(bad, msg):
    with pytest.raises(ValueError, match=msg):
        model.validate(portfolio(**bad), POS)


def test_validate_reports_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        model.validate(portfolio().drop(columns="phase"), POS)
