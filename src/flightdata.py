
""" File to load, clean and process the data from the GFC flight computer
such that a comparison can be made.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.constants import G0, LAPSE_RATE, R_AIR

# Timestamps
WRAP_S = 2**32 / 1e6
HPA_TO_PA = 100.0
# Mean Earth radius [m] used for the local flat-Earth projection.
EARTH_RADIUS = 6_371_008.8
# Time window kept in the processed files, relative to launch [s].
FLIGHT_WINDOW = (-60.0, 300.0)


# Converting the timestamps into seconds
def unwrap_us(t_us):
  """Convert wrapped timestamps to monotonic seconds.

  Parameters:
  t_us: array
    Raw timestamps [us].

  Returns:
  numpy.ndarray
    Time since boot [s].
  """
  t = np.asarray(t_us, dtype=float) / 1e6
  # Using half a wrap period as a safe threshold
  jumps = np.diff(t) < -WRAP_S / 2
  n_wraps = np.concatenate([[0], np.cumsum(jumps)])
  return t + n_wraps * WRAP_S


# Opening the data and setting it up
def _find_header_row(path, first_field):
  """
  Function to return the line number of the column-header row such that it is not hardcoded

  """
  with open(path, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
      if line.split(",")[0].strip() == first_field:
        return i
  raise ValueError(f"no header row found")


def load_gfc_baro(path):
  """Loading the GFC barometer data.

  Parameters:
  ----------
  path: str
    Path to BARO.CSV.

  Returns:
  pandas.DataFrame (includes the following)
    time since boot [s]
    pressure [Pa]
    bay_temp [degC]
    alt_asl [m]

  """
  df = pd.read_csv(path, skiprows=_find_header_row(path, "Baro"), skipinitialspace=True)
  df.columns = ["sensor", "t_us", "pressure_hpa", "bay_temp", "alt_asl"]
  return pd.DataFrame({
    "t": unwrap_us(df["t_us"].to_numpy()),
    "pressure": df["pressure_hpa"].to_numpy() * HPA_TO_PA,
    "bay_temp": df["bay_temp"].to_numpy(),
    "alt_asl": df["alt_asl"].to_numpy(),})


def load_gfc_accel(path):
  """Loading the GFC accelerometer data.

  Returns:
  pandas.DataFrame (includes the following)
    sensor (str)
    t [s]
    ax [m/s^2]
    ay [m/s^2]
    az [m/s^2]

  """

  df = pd.read_csv(path, skiprows=_find_header_row(path, "Accel"), skipinitialspace=True)
  df.columns = ["sensor", "t_us", "ax", "ay", "az"]
  df = df[df["sensor"].isin(["ACCEL1", "ACCEL2"])]

  # Unwrap using the logging order of the whole file
  t = unwrap_us(df["t_us"].to_numpy())
  return pd.DataFrame({
    "sensor": df["sensor"].to_numpy(),
    "t": t,
    "ax": df["ax"].to_numpy() * G0,
    "ay": df["ay"].to_numpy() * G0,
    "az": df["az"].to_numpy() * G0,})


def load_gfc_gps(path):
  """Loading the GFC GPS fixes

  Returns:
  pandas.DataFrame (includes the following)
    t [s]
    lat [deg]
    lon [deg]
    alt_gps [m]
    hdop
    fix
    sats

  """
  df = pd.read_csv(path, skiprows=_find_header_row(path, "GPS"), skipinitialspace=True)
  df.columns = ["sensor", "t_us", "lon", "lat", "alt_gps", "vdop", "hdop", "pdop",
                "fix", "sats", "utc"]
  out = pd.DataFrame({
    "t": unwrap_us(df["t_us"].to_numpy()),
    "lat": df["lat"].to_numpy(), "lon": df["lon"].to_numpy(),
    "alt_gps": df["alt_gps"].to_numpy(), "hdop": df["hdop"].to_numpy(),
    "fix": df["fix"].to_numpy(), "sats": df["sats"].to_numpy(),})
  return out[out["fix"] == 3].reset_index(drop=True)


def load_gfc_state(path):
  """Event times logged by the GFC state machine.

  Returns:
  dict (includes the following)
    launch [s]
    burnout [s]
    drogue [s]
    main [s]
    landing [s]
    main_alt_asl [m]
  """
  df = pd.read_csv(path, skiprows=_find_header_row(path, "Timestamp(us)"), skipinitialspace=True)
  last = df.iloc[-1]
  return {
    "launch": last["Launch Time(ms)"] / 1000.0,
    "burnout": last["Burnout Time(ms)"] / 1000.0,
    "drogue": last["Drogue Deploy Time(ms)"] / 1000.0,
    "main": last["Main Deploy Time(ms)"] / 1000.0,
    "landing": last["Landing Time(ms)"] / 1000.0,
    "main_alt_asl": float(last["Main Deploy Altitude(m)"]),}


def _trim(df, t_launch):
  """Shifting time so launch is t = 0 and keep only the flight window."""
  df = df.copy()
  df["t"] = df["t"] - t_launch
  keep = (df["t"] >= FLIGHT_WINDOW[0]) & (df["t"] <= FLIGHT_WINDOW[1])
  return df[keep].reset_index(drop=True)


def make_processed_data(gfc_dir, out_dir):
  """Converting the raw GFC folder into the trimmed files committed to the repo.

  Writes ``gfc_baro.csv``, ``gfc_accel.csv``, ``gfc_gps.csv`` and
  ``events.json`` to ``out_dir``. All times are seconds after the launch
  time logged by the GFC; all values are SI.

  Parameters
  gfc_dir: str
    Folder containing BARO.CSV, ACCEL.CSV, GPS.CSV, STATE.CSV.
  out_dir : str
    Destination folder.

  Returns:
  dict
    The contents written to events.json.
  """
  os.makedirs(out_dir, exist_ok=True)
  state = load_gfc_state(os.path.join(gfc_dir, "STATE.CSV"))
  t0 = state["launch"]
  baro = _trim(load_gfc_baro(os.path.join(gfc_dir, "BARO.CSV")), t0)
  accel = _trim(load_gfc_accel(os.path.join(gfc_dir, "ACCEL.CSV")), t0)
  gps = _trim(load_gfc_gps(os.path.join(gfc_dir, "GPS.CSV")), t0)

  pad = baro[baro["t"] < -1.0]                              # the minute before launch
  baro["alt_agl_device"] = baro["alt_asl"] - pad["alt_asl"].mean()
  gps_pad = gps[(gps["t"] < -1.0) & (gps["t"] > -30.0)]

  baro.to_csv(os.path.join(out_dir, "gfc_baro.csv"), index=False, float_format="%.4f")
  accel.to_csv(os.path.join(out_dir, "gfc_accel.csv"), index=False, float_format="%.4f")
  gps.to_csv(os.path.join(out_dir, "gfc_gps.csv"), index=False, float_format="%.7f")

  events = {
    "description": "GFC state-machine event times in s after launch; pad reference values.",
    "logged": {k: round(state[k] - t0, 3) for k in ("burnout", "drogue", "main", "landing")},
    "main_alt_agl_device_m": round(state["main_alt_asl"] - float(pad["alt_asl"].mean()), 2),
    "pad_pressure_pa": round(float(pad["pressure"].mean()), 1),
    "pad_lat": round(float(gps_pad["lat"].mean()), 7),
    "pad_lon": round(float(gps_pad["lon"].mean()), 7),
    "pad_gps_alt_m": round(float(gps_pad["alt_gps"].mean()), 2),}
  with open(os.path.join(out_dir, "events.json"), "w") as f:
    json.dump(events, f, indent=2)
  return events


def load_processed(data_dir):
  """Loading  the processed files.

  Returns:
  baro, accel, gps: pandas.DataFrame
  events: dict
  """
  baro = pd.read_csv(os.path.join(data_dir, "gfc_baro.csv"))
  accel = pd.read_csv(os.path.join(data_dir, "gfc_accel.csv"))
  gps = pd.read_csv(os.path.join(data_dir, "gfc_gps.csv"))
  with open(os.path.join(data_dir, "events.json")) as f:
    events = json.load(f)
  return baro, accel, gps, events


# Processing the data
def altitude_from_pressure(p, p0, T0):
  """Height above the pad from measured pressure, and the ISA.

  Parameters:
  p: float or array_like
    Measured pressure [Pa].
  p0: float
    The sensor's pad pressure [Pa].
  T0: float
    Pad air temperature [K].

  Returns:
  float
    Height above the pad [m].
  """
  p = np.asarray(p, dtype=float)
  return (T0 / LAPSE_RATE) * (1.0 - (p / p0) ** (R_AIR * LAPSE_RATE / G0))


def latlon_to_en(lat, lon, lat0, lon0):
  """Latitude and Longitude to East and North from a reference point.

  Returns:
  east, north: numpy.ndarray
    Offsets from (lat0, lon0) [m].
  """
  lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
  east = EARTH_RADIUS * np.cos(np.radians(lat0)) * np.radians(lon - lon0)
  north = EARTH_RADIUS * np.radians(lat - lat0)
  return east, north


def resample_uniform(t, x, dt, t_start=None, t_end=None):
  """
  Linearly interpolating samples onto a uniform time grid for smoothing.

  Returns:
  t_uniform, x_uniform: numpy.ndarray
  """
  t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
  t_start = t[0] if t_start is None else t_start
  t_end = t[-1] if t_end is None else t_end
  t_uniform = np.arange(t_start, t_end + dt / 2, dt)
  return t_uniform, np.interp(t_uniform, t, x)


def linear_fit_slope_kernel(n_points, dt):
  """Weights such that sum(w_j h_j) is the least-squares slope over a window.

  """
  if n_points < 3 or n_points % 2 == 0:
    raise ValueError(f"n_points must be odd and >= 3, got {n_points}")
  offsets = (np.arange(n_points) - (n_points - 1) / 2) * dt
  return offsets / np.sum(offsets**2)


def descent_rate(t, h, method="linefit", dt=0.05, window_s=2.0):
  """Vertical velocity from an altitude time series.

  Parameters:
  t, h: array_like
    Time [s] and altitude [m].
  method: {"central", "savgol", "linefit"}
    central: central differences with no smoothing.
    savgol: Savitzky-Golay, derivative of a moving quadratic fit.
    linefit: slope of a moving least-squares straight line.
  dt: float
    Resampling interval [s].
  window_s: float
    Smoothing window [s] for savgol and linefit.

  Returns:
  t_uniform: numpy.ndarray
  v : numpy.ndarray
    Vertical velocity [m/s], positive up.
  """
  t_u, h_u = resample_uniform(t, h, dt)
  n = int(round(window_s / dt)) | 1
  if method == "central":
    return t_u, np.gradient(h_u, dt)
  if method == "savgol":
    return t_u, savgol_filter(h_u, window_length=n, polyorder=2, deriv=1, delta=dt)
  if method == "linefit":
    w = linear_fit_slope_kernel(n, dt)
    v = np.convolve(h_u, w[::-1], mode="same")
    v[: n // 2] = np.nan
    v[-(n // 2):] = np.nan
    return t_u, v


# Showing the events
def _first_sustained(t, condition, hold_s, start_index=0):
  """Start time of the first run where condition holds."""
  run_start = None
  for i in range(start_index, len(t)):
    if condition[i]:
      if run_start is None:
        run_start = i
      if t[i] - t[run_start] >= hold_s:
        return float(t[run_start])
    else:
      run_start = None
  return None


def detect_launch(t, accel, threshold=3 * G0, hold_s=0.1):
    """Launch time defined as first moment the acceleration stays above the threshold.

    Parameters:
    t: array
      Time [s].
    accel: array
      Acceleration magnitude [m/s^2].
    threshold: float
      Default 3 g.
    hold_s: float
      Required duration [s].

    """
    t, accel = np.asarray(t, dtype=float), np.asarray(accel, dtype=float)
    t_launch = _first_sustained(t, accel > threshold, hold_s)
    return t_launch


def detect_apogee(t, h, smooth_s=1.0):
  """Time and height of the maximum of the smoothed altitude.

  Returns
  -------
  t_apogee, h_apogee: float
  """
  t, h = np.asarray(t, dtype=float), np.asarray(h, dtype=float)
  dt = np.median(np.diff(t))
  n = max(1, int(round(smooth_s / dt))) | 1
  h_smooth = np.convolve(h, np.ones(n) / n, mode="same")
  if n > 1:
    h_smooth[: n // 2] = -np.inf
    h_smooth[-(n // 2):] = -np.inf
  i = int(np.argmax(h_smooth))
  return float(t[i]), float(h_smooth[i])


def detect_landing(t, h, t_after, tol=3.0, hold_s=5.0):
  """Landing time

  """
  t, h = np.asarray(t, dtype=float), np.asarray(h, dtype=float)
  ground = np.median(h[-max(10, len(h) // 50):])
  start = int(np.searchsorted(t, t_after))
  t_land = _first_sustained(t, np.abs(h - ground) < tol, hold_s, start_index=start)
  return t_land
