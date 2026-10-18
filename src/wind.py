
"""
Model for the wind where it is horizontal and steady in time
"""

from dataclasses import dataclass, field

import numpy as np

# Getting wind within its east and north components
def wind_components(speed, direction_from_deg):
  """
  Function: Convert speed and meteorological direction to (u_east, v_north).

  Parameters:
  speed: float
    Wind speed [m/s]
  direction_from_deg: float
    Direction the wind blows FROM, degrees clockwise from north.

  Returns:
  numpy.ndarray
    Array [u_east, v_north] in m/s.

  Raises:
  ValueError
    If speed is negative or NaN.
  """
  if not (speed >= 0):
    raise ValueError(f"speed must be non-negative, got {speed}")

  rad = np.radians(direction_from_deg)
  u_east = -speed * np.sin(rad)
  v_north = -speed * np.cos(rad)

  return np.array([u_east, v_north], dtype=float)


# Class for having the same wind at every altitude
@dataclass(frozen=True)
class ConstantWind:
  """
  Class: The same wind at every altitude.

  Parameters:
  u_east, v_north: float
    Wind components [m/s].
  """

  u_east: float = 0.0
  v_north: float = 0.0

  def velocity(self, h):
    return np.array([self.u_east, self.v_north], dtype=float)


# Class for wind speed growing with height, but remaining in a constant direction
@dataclass(frozen=True)
class PowerLawWind:
  """
  Class: Wind speed growing with height as a power law; direction constant.
  This is a simple baseline profile for when only surface wind measurements are available.

  Parameters:
  speed_ref: float
    Measured wind speed [m/s] at height h_ref.
  h_ref: float
    Height of the measurement above the ground [m].
  direction_from_deg: float
    Meteorological direction in degrees.
  alpha: float
    Power-law exponent. Given to be about 1/7 for open terrain.

  """

  speed_ref: float
  h_ref: float = 10.0
  direction_from_deg: float = 270.0
  alpha: float = 1.0 / 7.0

  def __post_init__(self):
    if not (self.speed_ref >= 0):
        raise ValueError(f"speed_ref must be non-negative, got {self.speed_ref}")
    if not (self.h_ref > 0):
        raise ValueError(f"h_ref must be positive, got {self.h_ref}")

  def velocity(self, h):
    h_eff = np.maximum(h, self.h_ref)
    speed = self.speed_ref * (h_eff / self.h_ref) ** self.alpha
    return wind_components(speed, self.direction_from_deg)


# Class for when wind data is available from an interpolated table.
@dataclass(frozen=True)
class ProfileWind:
  """
  Class: Wind interpolated from a table of heights and components. Using linear interpolation

  Parameters:
  heights: array_like
    Heights above the pad [m].
  u_east, v_north: array_like
    Wind components [m/s] at each height.
  """

  heights: np.ndarray
  u_east: np.ndarray
  v_north: np.ndarray

  def __post_init__(self):
    for name in ("heights", "u_east", "v_north"):
      object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
    h, u, v = self.heights, self.u_east, self.v_north

    if h.ndim != 1 or u.ndim != 1 or v.ndim != 1:
      raise ValueError("heights, u_east, and v_north must all be 1-dimensional arrays")

    if not (len(h) == len(u) == len(v)):
      raise ValueError(
        f"heights, u_east, and v_north must have the same length: "
        f"got {len(h)}, {len(u)}, and {len(v)}")

    if len(h) < 2:
      raise ValueError(f"ProfileWind requires at least 2 points for interpolation, got {len(h)}")

    if np.any(np.isnan(h)) or np.any(np.isnan(u)) or np.any(np.isnan(v)):
      raise ValueError("ProfileWind inputs cannot contain NaN values")

    if not np.all(np.diff(h) > 0):
      raise ValueError("heights must be strictly increasing")

  def velocity(self, h):
    u = np.interp(h, self.heights, self.u_east)
    v = np.interp(h, self.heights, self.v_north)
    return np.array([float(u), float(v)], dtype=float)
