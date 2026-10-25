
"""
File to predict landing drift using a baseline hand-calculation, a wind profile estimation and error metrics.

"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.analytical import terminal_velocity
from src.constants import G0
from src.descent import descent_rhs
from src.wind import ConstantWind, ProfileWind



def bearing_deg(east, north):
  """
  Function to get the bearing of a displacement, degrees clockwise from North.

  """
  return float(np.degrees(np.arctan2(east, north)) % 360)


def hand_calc_drift(apogee_alt, main_alt, v_drogue, v_main, wind_uv):
  """
  Function for the baseline drift estimate where each descent stage is at a constant speed.

  Parameters:
  apogee_alt, main_alt: float
    Heights above the pad [m].
  v_drogue, v_main: float
    Descent speeds [m/s], positive.
  wind_uv: array_like
    Wind [m/s].

  Returns:
  drift_en: numpy.ndarray
    Displacement from the apogee point [m].
  t_descent: float
    Total descent time [s].
  """
  t_descent = (apogee_alt - main_alt) / v_drogue + main_alt / v_main
  return np.asarray(wind_uv, dtype=float) * t_descent, float(t_descent)


def gps_wind_profile(t, east, north, alt, bin_m=250.0, min_points=5):
  """
  Function to estimate the wind profile from the GPS track of a parachute descent

  Parameters:
  t: array_like
    GPS fix times [s].
  east, north: array_like
    Positions [m].
  alt: array_like
    Altitude above the pad at the fix times [m]
  bin_m: float
      Altitude bin height [m].
  min_points : int
      Skip bins with fewer fixes than this (GPS gaps).

  Returns:
  ProfileWind
    Wind components at the mean altitude of each usable bin.
  """
  t, east, north, alt = (np.asarray(a, dtype=float) for a in (t, east, north, alt))
  heights, us, vs = [], [], []
  edges = np.arange(0.0, alt.max() + bin_m, bin_m)
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (alt >= lo) & (alt < hi)
    if m.sum() < min_points:
        continue
    heights.append(alt[m].mean())
    us.append(np.polyfit(t[m], east[m], 1)[0])
    vs.append(np.polyfit(t[m], north[m], 1)[0])
  return ProfileWind(heights=heights, u_east=us, v_north=vs)


def landing_error(predicted_en, measured_en):
  """
  Function for the distance and bearing error between predicted and measured landing points.

  Returns:
  dict (containing the following)
    distance between the points in m, 
    bearing error in deg.
    range error in m.
  """
  p, m = np.asarray(predicted_en, dtype=float), np.asarray(measured_en, dtype=float)
  d_bearing = (bearing_deg(*p) - bearing_deg(*m) + 180.0) % 360.0 - 180.0
  return {
    "distance": float(np.hypot(*(p - m))),
    "bearing_error": float(d_bearing),
    "range_error": float(np.hypot(*p) - np.hypot(*m)),}

