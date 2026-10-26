

"""
File for the exact solutions used to verify the numerical code for a simplified problem.

"""

import numpy as np

from src.constants import G0

# Function to calculate the terminal velocity
def terminal_velocity(mass, cds, rho, g=G0):
  """
  Function: Calculate the steady descent speed where drag balances weight (terminal velocity)

  Parameters:
  mass: float
    Suspended mass [kg]. Must be positive.
  cds: float
    Drag area Cd * S [m^2]. Must be positive.
  rho: float
    Air density [kg/m^3]. Must be positive.
  g: float, optional
    Gravitational acceleration [m/s^2]. Default: G0.

  Returns:
  float:
    Terminal speed [m/s] (positive number, direction is down).

  Raises:
  ValueError:
    If mass, cds, rho, or g is not positive.
  """
  # Checking that the physical parameters are positive for validity
  if not (mass > 0):
    raise ValueError(f"mass must be positive")
  if not (cds > 0):
    raise ValueError(f"cds must be positive")
  if not (rho > 0):
    raise ValueError(f"rho must be positive")
  if not (g > 0):
    raise ValueError(f"g must be positive")

  return float(np.sqrt(2.0 * mass * g / (rho * cds)))


# Function for the speed and distance
def analytic_fall(t, mass, cds, rho, g=G0):
  """
  Function: Calculate the exact speed and distance fallen for a drag-limited fall from rest.

  Parameters:
  t: float or numpy.ndarray
    Time since release [s], t >= 0.
  mass: float
    Suspended mass [kg]. Must be positive.
  cds: float
    Drag area Cd * S [m^2]. Must be positive.
  rho: float
    Air density [kg/m^3]. Must be positive.
  g: float, optional
    Gravitational acceleration [m/s^2]. Default: G0.

  Returns:
  speed_down: float or numpy.ndarray
    Downward speed [m/s].
  drop: float or numpy.ndarray
    Distance fallen [m] (positive).

  Raises:
  ValueError
    If any time value in t is negative.
  """
  # Calculating the terminal velocity
  v_t = terminal_velocity(mass, cds, rho, g)

  # Converting the input time to a NumPy FP array
  t = np.asarray(t, dtype=float)

  # Rejecting negative times
  if np.any(t < 0):
    raise ValueError(f"t must be non-negative")

  # Defining a dimensionless time variable for a simplied analytical expression for velocity and distance
  x = g * t / v_t

  # Downward speed
  speed_down = v_t * np.tanh(x)

  # Distance fallen from the release point
  drop = (v_t**2 / g) * (np.logaddexp(x, -x) - np.log(2.0))
  if t.ndim == 0:
    return float(speed_down), float(drop)
  return speed_down, drop
