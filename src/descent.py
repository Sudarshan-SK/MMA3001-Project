
"""
Completing the 3D point-mass parachute descent model.

"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from atmosphere import ISAAtmosphere
from constants import G0
from wind import ConstantWind

# Class to describe the parachute
@dataclass(frozen=True)
class Parachute:
  """
  Class: A parachute described by its nominal diameter and drag coefficient.

  Parameters:
  name: str
    Label used in plots.
  diameter: float
    Nominal diameter [m].
  cd: float
    Drag coefficient referenced to the nominal area
  """

  name: str
  diameter: float
  cd: float

  def __post_init__(self):
    if not (self.diameter > 0):
      raise ValueError(f"diameter must be positive, got {self.diameter}")
    if not (self.cd > 0):
      raise ValueError(f"cd must be positive, got {self.cd}")

  @property
  def cds(self):
    return self.cd * (np.pi * self.diameter**2 / 4.0)


# Class for the information needed to simulate
@dataclass(frozen=True)
class DescentConfig:
  """
  Class: Everything the descent simulation needs, in one object.

  Parameters:
  mass: float
    Total descending mass [kg].
  drogue, main: Parachute
    The two stages.
  main_deploy_alt: float
    Main deployment altitude above the pad [m].
  atmosphere: object with density(h)
    Standard ISA.
  wind: object with velocity(h)
      No wind case considered
  g: float
    Gravitational acceleration [m/s^2].
  """

  mass: float
  drogue: Parachute
  main: Parachute
  main_deploy_alt: float
  atmosphere: object = field(default_factory=ISAAtmosphere)
  wind: object = field(default_factory=ConstantWind)
  g: float = G0

  def __post_init__(self):
    if not (self.mass > 0):
      raise ValueError(f"mass must be positive, got {self.mass}")
    if not (self.main_deploy_alt >= 0):
      raise ValueError(f"main_deploy_alt must be non-negative, got {self.main_deploy_alt}")


# Function for the RHS of the descent equation
def descent_rhs(t, y, mass, cds, atmosphere, wind, g=G0):
  """
  Function: Right-hand side dy/dt of the descent equations.

  Parameters:
  t: float
    Time [s].
  y: numpy.ndarray
    State [x, y, z, vx, vy, vz].
  mass: float
    Mass [kg].
  cds: float
    Drag area of the parachute currently deployed [m^2].
  atmosphere: object with density(h)
  wind: object with velocity(h)
  g: float
      Gravitational acceleration [m/s^2].

  Returns:
  numpy.ndarray
    [vx, vy, vz, ax, ay, az].
  """
  # Unpack the state. Slices are views (no copying), so this is cheap.
  z = y[2]
  vel = y[3:6]

  # Adaptive solvers evaluate trial stages that can land below the ground
  # before the ground event stops the integration (in uniform motion the step
  # size can grow to hundreds of seconds). The atmosphere model rightly refuses
  # altitudes far below the pad, so look up air properties at max(z, 0).
  # This only affects points the solver is about to discard. A good example
  # for "which part of your solution is vulnerable to numerical failure?"
  z_air = max(z, 0.0)

  rho = atmosphere.density(z_air)
  w = wind.velocity(z_air)
  wind_3d = np.array([w[0], w[1], 0.0], dtype=float)
  v_rel = vel - wind_3d

  v_rel_norm = np.linalg.norm(v_rel)
  drag_acc = -0.5 * rho * cds * v_rel_norm * v_rel / mass

  acc = drag_acc.copy()
  acc[2] -= g

  return np.concatenate([vel, acc])


def _altitude_crossing(target):
  """
  Making a solve_ivp event function that triggers when z falls through ``target``.
  """

  def event(t, y, *args):
    return y[2] - target

  event.terminal = True
  event.direction = -1
  return event



# Class for the results
@dataclass
class DescentResult:
  """
  Class: Output of the function simulate_descent.

  Attributes:
  t: numpy.ndarray
    Times [s], shape (n_points,).
  y: numpy.ndarray
    States, shape (n_points, 6).
  t_main: float or None
    Time the main deployed [s].
  t_land: float or None
    Time of touchdown [s].
  landing_xy : numpy.ndarray or None
    Landing position [east, north] [m] relative to the pad.
  nfev: int
    Total right-hand-side evaluations.
  """

  t: np.ndarray
  y: np.ndarray
  t_main: float | None
  t_land: float | None
  landing_xy: np.ndarray | None
  nfev: int


# Function to simulate the descent
def simulate_descent(config, y0, t0=0.0, t_max=2000.0, method="RK45",
                     rtol=1e-8, atol=1e-8, max_step=np.inf):
  """
  Function: Simulate a two-stage (drogue then main) descent to the ground.

  Parameters:
  config: DescentConfig
  y0: array_like
    Initial state [x, y, z, vx, vy, vz].
  t0: float
    Start time [s].
  t_max: float
    Give up after this time [s] if the ground has not been reached.
  method: str
    Any solve_ivp method ("RK45", "DOP853", "Radau", "BDF", "LSODA").
  rtol, atol: float
    Solver tolerances.
  max_step: float
    Largest step the solver may take [s].

  Returns:
  DescentResult

  Raises:
  ValueError
      If the starting altitude is below the ground (z < 0).
  """
  y0 = np.asarray(y0, dtype=float)
  if y0[2] < 0:
    raise ValueError(f"starting altitude must be >= 0, got {y0[2]}")

  ground = _altitude_crossing(0.0)
  deploy = _altitude_crossing(config.main_deploy_alt)
  solver_kw = dict(method=method, rtol=rtol, atol=atol, max_step=max_step)

  t_parts, y_parts, nfev = [], [], 0
  t_main = None
  t_start, y_start = t0, y0


  # Drogue descent
  if y0[2] > config.main_deploy_alt:
    sol = solve_ivp(
      descent_rhs,
      (t_start, t_max),
      y_start,
      args=(config.mass, config.drogue.cds, config.atmosphere, config.wind, config.g),
      events=[deploy, ground],
      **solver_kw,)
    t_parts.append(sol.t)
    y_parts.append(sol.y.T)
    nfev += sol.nfev

    if len(sol.t_events[0]) > 0:
      t_main = float(sol.t_events[0][0])
      t_start = t_main
      y_start = sol.y_events[0][0]
    else:
      return _assemble(t_parts, y_parts, t_main, sol.t_events[1], nfev)

  # Mains descent
  sol = solve_ivp(
    descent_rhs,
    (t_start, t_max),
    y_start,
    args=(config.mass, config.main.cds, config.atmosphere, config.wind, config.g),
    events=[ground],
    **solver_kw,)
  t_parts.append(sol.t)
  y_parts.append(sol.y.T)
  nfev += sol.nfev

  return _assemble(t_parts, y_parts, t_main, sol.t_events[0], nfev)


def _assemble(t_parts, y_parts, t_main, ground_events, nfev):

  for i in range(1, len(t_parts)):
    t_parts[i], y_parts[i] = t_parts[i][1:], y_parts[i][1:]
  t = np.concatenate(t_parts)
  y = np.vstack(y_parts)
  landed = len(ground_events) > 0
  return DescentResult(
    t=t,
    y=y,
    t_main=t_main,
    t_land=float(ground_events[0]) if landed else None,
    landing_xy=y[-1, 0:2].copy() if landed else None,
    nfev=nfev,)
