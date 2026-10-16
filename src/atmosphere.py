

"""Air density models used  to simulate the parachute descent.

The atmosphere is defined by a density (rho). A constant-density model, and an ISA model is used and compared.

Convention:
h = the height above the launch pad in metres (AGL)
p0 = the measured ground pressure 
T0 = temperature on the day

This is done so that h = 0 is the pad, not sea level
"""

from dataclasses import dataclass

import numpy as np

from constants import G0, R_AIR, LAPSE_RATE, P0_STD, T0_STD, H_TROPOPAUSE


# Defining the lowest altitude possible, and is slightly negative so errors are not raised
H_MIN = -500.0


# Function to raise an error if any altitude is not allowed
def _check_altitude(h, h_max):
  """
  Function:
  Raise a ValueError if any altitude is NaN or outside of the range given by [H_MIN, h_max].

  Parameters:
  h: float
    Altitude above the pad in metres.
  h_max: float
    Highest valid altitude above the pad in metres.

  Raises:
  ValueError
    If any value is NaN, below H_MIN or above h_max.
  """

  h_arr = np.asarray(h, dtype=float)

  # Raising error if altitude is NaN
  if np.any(np.isnan(h_arr)):
    raise ValueError(f"altitude contains NaN: {h}")

  # Raising error if altitude is below the minimum
  if np.any(h_arr < H_MIN):
    min_val = np.min(h_arr)
    raise ValueError(f"altitude below minimum bound H_MIN ({H_MIN} m): got {min_val}")

  # Raising error if altitude is above the maximum
  if np.any(h_arr > h_max):
    max_val = np.max(h_arr)
    raise ValueError(f"altitude above maximum bound h_max ({h_max} m): got {max_val}")


# Function to define the constant atmosphere model
@dataclass(frozen=True)
class ConstantAtmosphere:
  """
  Function:
  Define a constant air density at every altitude. This is so that a simple baseline model can be compared, and to ensure the descent ODE has an exact solution for a constant density.

  Parameters:
  rho: float
    Air density in kg/m^3
  """
  
  # Defining the air density value
  rho = 1.225

  # Ensuring air density is possitive
  def __post_init__(self):
    if not (self.rho > 0):
      raise ValueError(f"rho must be positive, got {self.rho}")

  # Returning the value for the air density with the same shape as the altitude
  def density(self, h):
    h_arr = np.asarray(h, dtype=float)
    _check_altitude(h_arr, H_TROPOPAUSE)
    out = np.full_like(h_arr, self.rho)
    return float(out) if out.ndim == 0 else out

# Function for an atmosphere model that varies with altitude. Using the ISA model
@dataclass(frozen=True)
class ISAAtmosphere:
  """
  Function:
  Defining an International Standard Atmosphere that is anchored at the pad. This includes 
  International Standard Atmosphere (troposphere) anchored at the pad. This model is used until an altitude of 11km

  Convention:
  T(h) = T0 - L h
    Temperature falls linearly with altitude
  p(h) = p0 (T(h) / T0) ** (g0 / (R L))
    Pressure follows from hydrostatic balance plus ideal gas law
  rho(h) = p(h) / (R T(h))
    Density from the ideal gas law

  Parameters:
  p0: float
    Pressure at the pad in Pa.
  T0: float
    Air temperature at the pad in K
  """

  # Defining parameters at ground level
  p0 = P0_STD
  T0 = T0_STD

  # Ensuring values are positive and valid.
  def __post_init__(self):
    if not (self.p0 > 0):
      raise ValueError(f"p0 must be positive and non-NaN, got {self.p0}")
    if not (self.T0 > 0):
      raise ValueError(f"T0 must be positive and non-NaN, got {self.T0}")

  # Defining the air temperature in K at an altitude h above the pad in m.
  def temperature(self, h):
    _check_altitude(h, H_TROPOPAUSE)
    h_arr = np.asarray(h, dtype=float)
    t = self.T0 - LAPSE_RATE * h_arr
    return float(t) if t.ndim == 0 else t

  # Defining the air pressure in Pa at an altitude h above the pad in m.
  def pressure(self, h):
    t = self.temperature(h)
    exponent = G0 / (R_AIR * LAPSE_RATE)
    p = self.p0 * (t / self.T0) ** exponent
    return float(p) if isinstance(p, np.ndarray) and p.ndim == 0 else p

  # Defining the air density in kg/m^3 at an altitude h above the pad in m.
  def density(self, h):
    t = self.temperature(h)
    p = self.pressure(h)
    rho = p / (R_AIR * t)
    return float(rho) if isinstance(rho, np.ndarray) and rho.ndim == 0 else rho
