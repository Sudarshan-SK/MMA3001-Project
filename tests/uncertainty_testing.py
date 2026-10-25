
# File for testing uncertainty with the descent model


import numpy as np
import pytest

from src.atmosphere import ConstantAtmosphere, ISAAtmosphere
from src.uncertainty import (ScaledRotatedWind, UncertaintySpec, inside_ellipse, landing_ellipse,
                             nominal_inputs, sample_inputs, simulate_descent_batch,
                             simulate_descent_loop, tornado, wind_uv_batch)
from src.wind import ConstantWind, PowerLawWind, ProfileWind

ATM = ISAAtmosphere(p0=99438.2, T0=300.0)
WIND = ProfileWind(heights=[100, 1000, 2000, 3500], u_east=[6, 6, 10, 16], v_north=[4, 3, 5, 2])
SPEC = UncertaintySpec(mass=20.0, cds_drogue=0.85, cds_main=6.7, main_alt=456.0,
                       apogee_alt=3500.0, sigma_cds_drogue=0.01, sigma_cds_main=0.1)


# Testing if the vectorised RK4 matches random samples
def test_batch_matches_trusted_simulator():
  s = sample_inputs(SPEC, 12, seed=3)
  batch = simulate_descent_batch(s, ATM, WIND, h=0.1)
  loop = simulate_descent_loop(s, ATM, WIND, rtol=1e-9, atol=1e-9)
  miss = np.hypot(*(batch["landing_xy"] - loop["landing_xy"]).T)
  assert miss.max() < 1.0
  np.testing.assert_allclose(batch["t_land"], loop["t_land"], atol=0.1)


# Testing how error shrinks with the increase in step size
def test_batch_error_shrinks_with_step_size():
  s = nominal_inputs(SPEC)
  ref = simulate_descent_loop(s, ATM, WIND, rtol=1e-10, atol=1e-10)["landing_xy"][0]
  errs = [np.hypot(*(simulate_descent_batch(s, ATM, WIND, h=h)["landing_xy"][0] - ref))
          for h in (0.4, 0.1, 0.025)]
  assert errs[0] > errs[1] > errs[2]

# Testing if starting at rest in constant wind affects the rocket
def test_drift_is_wind_times_time_in_constant_wind_and_density():
    s = nominal_inputs(SPEC)
    out = simulate_descent_batch(s, ConstantAtmosphere(1.1), ConstantWind(5.0, 0.0), h=0.05)
    x, t = out["landing_xy"][0, 0], out["t_land"][0]
    assert 5.0 * (t - 5.0) < x < 5.0 * t

# Testing the sampling rate
def test_sampling_is_reproducible_and_spread_is_right():
    a, b = sample_inputs(SPEC, 20000, seed=7), sample_inputs(SPEC, 20000, seed=7)
    np.testing.assert_array_equal(a["mass"], b["mass"])
    assert np.std(a["mass"]) == pytest.approx(0.03 * 20.0, rel=0.03)
    assert np.std(a["wind_rot_deg"]) == pytest.approx(10.0, rel=0.03)

# Testing the wind location and scaling with north wind turned east
def test_wind_rotation_and_scaling():
    w = ScaledRotatedWind(ConstantWind(0.0, 10.0), scale=0.5, rot_deg=90.0)
    np.testing.assert_allclose(w.velocity(100.0), [5.0, 0.0], atol=1e-12)


def test_wind_batch_matches_scalar_methods():
    z = np.array([5.0, 50.0, 500.0, 2500.0])
    for wind in (WIND, ConstantWind(3.0, -2.0), PowerLawWind(6.0, 2.0, 240.0)):
        u, v = wind_uv_batch(wind, z)
        scalar = np.array([wind.velocity(h) for h in z])
        np.testing.assert_allclose(np.column_stack([u, v]), scalar, atol=1e-12)

# Testing the recovery zone ellipse
def test_ellipse_recovers_known_gaussian():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.normal(0, 30, 200000), rng.normal(0, 10, 200000)])
    ell = landing_ellipse(pts)
    assert ell["semi_major"] == pytest.approx(np.sqrt(5.991) * 30, rel=0.01)
    assert ell["semi_minor"] == pytest.approx(np.sqrt(5.991) * 10, rel=0.01)
    assert ell["bearing_major_deg"] == pytest.approx(90.0, abs=1.0)
    inside = np.mean([inside_ellipse(p, ell) for p in pts[:2000]])
    assert inside == pytest.approx(0.95, abs=0.015)


def test_ellipse_rejects_bad_shape():
    with pytest.raises(ValueError):
        landing_ellipse(np.zeros((2, 2)))

# Testing the tornado plot on a linear model
def test_tornado_on_linear_model():
    rows = tornado(lambda a, b: 3 * a + b, {"a": 1.0, "b": 1.0}, {"a": 0.1, "b": 0.1})
    assert rows[0]["parameter"] == "a"
    assert rows[0]["swing"] == pytest.approx(0.6)
    assert rows[1]["swing"] == pytest.approx(0.2)


def test_spec_rejects_bad_values():
    with pytest.raises(ValueError):
        UncertaintySpec(mass=20, cds_drogue=0.85, cds_main=6.7, main_alt=4000, apogee_alt=3500)


