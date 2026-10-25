
# File for testing the flight data and drift


import numpy as np
import pytest
from scipy.signal import savgol_filter

from src.analytical import terminal_velocity
from src.atmosphere import ConstantAtmosphere, ISAAtmosphere
from src.constants import G0
from src.flightdata import (WRAP_S, altitude_from_pressure, descent_rate, detect_apogee,
                            detect_landing, detect_launch, latlon_to_en, linear_fit_slope_kernel,
                            load_gfc_accel, load_gfc_baro, load_gfc_state, resample_uniform,
                            unwrap_us)
from src.drift import (bearing_deg,
                          gps_wind_profile, hand_calc_drift, landing_error)


# Loading
BARO_TEXT = """Barometer data:
Baro: MS5607

Baro,Timestamp(us),Pressure(Pa),Temperature(degC),Altitude ASL(m)
BARO1, 4294000000, 994.380, 32.380, 158.299
BARO1, 100000, 994.340, 32.370, 158.637
BARO1, 200000, 994.320, 32.380, 158.807
"""

ACCEL_TEXT = """Accelerometer data:
Accel 1: BMI088
Accel 2: ASM330
Accel 3: H3LIS100

Accel,Timestamp(us),accX(g),accY(g),accZ(g)
ACCEL3, 654085, 0.000, -0.780, -0.780
ACCEL2, 675899, 0.030, 0.047, -0.761
ACCEL1, 677953, -0.026, 0.126, -1.000
"""

def write(tmp_path, name, text):
  p = tmp_path / name
  p.write_text(text)
  return str(p)


def test_unwrap_handles_two_rollovers():
  raw = np.array([4.29e9, 1e6, 4.29e9, 2e6])
  t = unwrap_us(raw)
  assert np.all(np.diff(t) > 0)
  assert t[3] == pytest.approx(2.0 + 2 * WRAP_S)


def test_unwrap_leaves_monotonic_data_alone():
  raw = np.array([0, 1e6, 2e6])
  np.testing.assert_allclose(unwrap_us(raw), [0.0, 1.0, 2.0])


def test_baro_converts_hpa_and_unwraps(tmp_path):
  df = load_gfc_baro(write(tmp_path, "BARO.CSV", BARO_TEXT))
  assert df["pressure"].iloc[0] == pytest.approx(99438.0)
  assert np.all(np.diff(df["t"]) > 0)
  assert df["t"].iloc[1] == pytest.approx(0.1 + WRAP_S)


def test_accel_drops_accel3_and_converts_g(tmp_path):
  df = load_gfc_accel(write(tmp_path, "ACCEL.CSV", ACCEL_TEXT))
  assert set(df["sensor"]) == {"ACCEL1", "ACCEL2"}
  assert df.loc[df["sensor"] == "ACCEL1", "az"].iloc[0] == pytest.approx(-G0)


def test_state_times_in_seconds(tmp_path):
  text = ("Flight State Data:\n\n"
          "Timestamp(us),Flight State,Up Axis,Drogue Device Status,Mains Device Status,"
          "Launch Time(ms),Burnout Time(ms),Drogue Deploy Time(ms),Main Deploy Time(ms),"
          "Landing Time(ms),Starting Altitude(m),Starting Pressure(Pa),Starting Temperature(degC),"
          "Burnout Altitude(m),Drogue Deploy Altitude(m),Main Deploy Altitude(m),"
          "Landing Altitude(m),Available Flash Memory (kB)\n"
          "1, LANDED, X, GOOD, BAD, 8547784, 8554886, 8573388, 8716804, 8784142, "
          "158.64, 994.34, 32.41, 1721.34, 3587.62, 614.36, 156.52, 7571669.00\n")
  s = load_gfc_state(write(tmp_path, "STATE.CSV", text))
  assert s["launch"] == pytest.approx(8547.784)
  assert s["main_alt_asl"] == pytest.approx(614.36)

  # Drifting under parachutes
MASS = 33.0
ATM = ISAAtmosphere(p0=99434.0, T0=300.0)
CDS_TRUE = 0.90

# Testing the bearing compass points
def test_bearing_compass_points():
  assert bearing_deg(0, 1) == 0.0
  assert bearing_deg(1, 0) == 90.0
  assert bearing_deg(0, -1) == 180.0
  assert bearing_deg(-1, 0) == 270.0

# Testing the GPS wind data
def test_gps_wind_recovers_constant_drift():
  t = np.arange(0.0, 150.0, 1.0)
  alt = 3000.0 - 20.0 * t
  wind = gps_wind_profile(t, 7.0 * t, -3.0 * t, alt)
  np.testing.assert_allclose(wind.u_east, 7.0)
  np.testing.assert_allclose(wind.v_north, -3.0)



