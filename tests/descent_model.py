
# File to test whether the descent model works


import numpy as np
import pytest

from src.analytical import analytic_fall, terminal_velocity
from src.atmosphere import ConstantAtmosphere, ISAAtmosphere
from src.descent import DescentConfig, Parachute, descent_rhs, simulate_descent
from src.integrators import euler_step, fixed_step_solve, rk4_step
from src.wind import ConstantWind, PowerLawWind, ProfileWind, wind_components


# Testing the atmosphere conditions

# Testing if the ISA value is correct at ground level
def test_isa_sea_level_density():
  atm = ISAAtmosphere()
  assert atm.density(0.0) == pytest.approx(1.225, rel=1e-3)

# Testing if the ISA table at altitude is correct.
def test_isa_density_at_3km():
  atm = ISAAtmosphere()
  assert atm.density(3000.0) == pytest.approx(0.909, rel=5e-3)

# Testing if temperature drops approximately 6.5 K per km
def test_isa_temperature_lapse():
  atm = ISAAtmosphere()
  assert atm.temperature(1000.0) == pytest.approx(288.15 - 6.5)

# Testing if the constant atmosphere model is actually constant
def test_constant_atmosphere_is_constant():
  atm = ConstantAtmosphere(rho=1.1)
  assert atm.density(0.0) == 1.1
  assert np.all(atm.density(np.array([0.0, 500.0, 3000.0])) == 1.1)


# Running tests based on the wind components
# Testing if speed is equal at a reference h
def test_power_law_at_reference_height():
  w = PowerLawWind(speed_ref=5.0, h_ref=10.0, direction_from_deg=270.0)
  np.testing.assert_allclose(w.velocity(10.0), [5.0, 0.0], atol=1e-12)


# Testing the wind linear interpolation
def test_profile_hits_table_values_exactly():
  w = ProfileWind(heights=[0, 1000, 3000], u_east=[2, 6, 9], v_north=[1, 3, -2])
  np.testing.assert_allclose(w.velocity(1000.0), [6.0, 3.0])


# Testing the integrators

# Testing a simple decay integrator
def decay(t, y):
  return -y

# Testing a one Euler step
def test_euler_single_step_by_hand():
  assert euler_step(decay, 0.0, np.array([1.0]), 0.1)[0] == pytest.approx(0.9)

# Testing one RK4 step and ensuring it equals the Taylor series expansion
def test_rk4_single_step_matches_taylor_series():
  h = 0.1
  taylor = 1 - h + h**2 / 2 - h**3 / 6 + h**4 / 24
  assert rk4_step(decay, 0.0, np.array([1.0]), h)[0] == pytest.approx(taylor, rel=1e-14)

# Testiing the solver ends at the specified point
def test_solver_ends_exactly_at_t_end():
  res = fixed_step_solve(decay, (0.0, 0.3), [1.0], 0.25)
  assert res.t[-1] == pytest.approx(0.3)
  assert len(res.t) == 3

# Creating a slope of log error vs log h
def observed_order(step, steps=(0.1, 0.05, 0.025, 0.0125)):
  errors = [abs(fixed_step_solve(decay, (0.0, 1.0), [1.0], h, step=step).y[-1, 0]
                - np.exp(-1.0)) for h in steps]
  slope, _ = np.polyfit(np.log(steps), np.log(errors), 1)
  return slope

# Testing the order of the euler and rk4
def test_euler_is_first_order():
  assert observed_order(euler_step) == pytest.approx(1.0, abs=0.1)

def test_rk4_is_fourth_order():
  assert observed_order(rk4_step) == pytest.approx(4.0, abs=0.2)


# Testing a simple analytical solution
MASS = 33.0
DROGUE = Parachute("drogue", 0.915, 1.3)
MAIN = Parachute("main", 2.44, 1.8)
RHO = 1.1
CALM = ConstantWind()
CONST_ATM = ConstantAtmosphere(RHO)

# Testing the RHS with drogue, constant density and no wind
def vertical_rhs(t, y):
  return descent_rhs(t, y, MASS, DROGUE.cds, CONST_ATM, CALM)

# Testing a fall from rest at 3000m
def fall_from_rest(h, t_end=10.0, step=rk4_step):
  y0 = [0.0, 0.0, 3000.0, 0.0, 0.0, 0.0]
  return fixed_step_solve(vertical_rhs, (0.0, t_end), y0, h, step=step).y[-1]

# Testing if RK4 with a small step converges
def test_rk4_matches_exact_tanh_solution():
  speed_exact, drop_exact = analytic_fall(10.0, MASS, DROGUE.cds, RHO)
  y = fall_from_rest(0.01)
  assert -y[5] == pytest.approx(speed_exact, rel=1e-8)
  assert 3000.0 - y[2] == pytest.approx(drop_exact, rel=1e-8)

# Testing the terminal velocity
def test_reaches_terminal_velocity():
  y = fall_from_rest(0.05, t_end=60.0)
  assert -y[5] == pytest.approx(terminal_velocity(MASS, DROGUE.cds, RHO), rel=1e-3)


# Testing that there is no horizontal motion when there is no wind
def test_no_wind_means_no_horizontal_motion():
  y = fall_from_rest(0.05, t_end=30.0)
  assert y[0] == 0.0 and y[1] == 0.0

# Testing during a constant wind scenario
def test_drifts_at_wind_speed_in_steady_wind():
  wind = ConstantWind(u_east=8.0, v_north=-3.0)
  rhs = lambda t, y: descent_rhs(t, y, MASS, DROGUE.cds, CONST_ATM, wind)
  y = fixed_step_solve(rhs, (0.0, 60.0), [0, 0, 3000, 0, 0, 0], 0.05).y[-1]
  np.testing.assert_allclose(y[3:5], [8.0, -3.0], rtol=1e-3)


# Creating a configuration to test mains and drogue descent
def make_config(wind=CALM, main_deploy_alt=450.0):
  return DescentConfig(mass=MASS, drogue=DROGUE, main=MAIN, main_deploy_alt=main_deploy_alt, atmosphere=CONST_ATM, wind=wind)

# Testing the event location
def test_main_deploys_at_exact_time_when_starting_at_terminal_speed():
  v_t = terminal_velocity(MASS, DROGUE.cds, RHO)
  res = simulate_descent(make_config(), [0, 0, 3000.0, 0, 0, -v_t])
  assert res.t_main == pytest.approx((3000.0 - 450.0) / v_t, rel=1e-6)

# Testing the terminal speed under mains
def test_lands_at_main_terminal_speed():
  v_t_main = terminal_velocity(MASS, MAIN.cds, RHO)
  res = simulate_descent(make_config(), [0, 0, 3000.0, 0, 0, 0])
  assert res.t_land is not None
  assert res.y[-1, 2] == pytest.approx(0.0, abs=1e-6)
  assert -res.y[-1, 5] == pytest.approx(v_t_main, rel=1e-3)

# Testing rocket drift
def test_drift_equals_wind_times_time_when_moving_with_the_wind():
  wind = ConstantWind(u_east=9.0, v_north=4.0)
  res = simulate_descent(make_config(wind), [0, 0, 3000.0, 9.0, 4.0, 0.0])
  np.testing.assert_allclose(res.landing_xy, [9.0 * res.t_land, 4.0 * res.t_land], rtol=1e-6)
