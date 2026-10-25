
"""
File to save the ODE integrations used in the simulation.
This is used to solve the following initial value problem with a constant step size of h
  dy/dt = f(t, y), y(t0) = y0
"""

from dataclasses import dataclass

import numpy as np

# Function to forward step through the Euler method
def euler_step(f, t, y, h, *args):
  """
  Function to advance one step with the forward (explicit) Euler method
  
  Parameters:
  f: callable
    f(t, y, ) -> dy/dt
  t: float
    current time in s.
  y: numpy.ndarray
    current state.
  h: float
    step size in s.
  args: extra arguments if needed
  
  Returns:
  numpy.ndarray
    state at t + h
  """
  return y + h * f(t, y, *args)

# Function to forward one step with the classical 4th-order Runge-Kutta method
def rk4_step(f, t, y, h, *args):
  """
  Function to Advance one step with the classical 4th-order Runge-Kutta method.
  Using a fourth-order with weights 1/6, 2/6, 2/6, 1/6.


  Parameters:
  f: callable
    f(t, y, ) -> dy/dt
  t: float
    current time in s.
  y: numpy.ndarray
    current state.
  h: float
    step size in s.
  args: extra arguments if needed
  
  Returns:
  numpy.ndarray
    state at t + h
  """
  k1 = f(t, y, *args)
  k2 = f(t + h / 2, y + h / 2 * k1, *args)
  k3 = f(t + h / 2, y + h / 2 * k2, *args)
  k4 = f(t + h, y + h * k3, *args)
  return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Number of RHS evaluations per step for each method.
STAGES = {euler_step: 1, rk4_step: 4}

# Class for the result of the fixed step solve
@dataclass
class FixedStepResult:
  """
  Output of function fixed_step_solve.

  Attributes:
  t: numpy.ndarray
  Times, shape (n_points,).
  y: numpy.ndarray
    States, shape (n_points, n_states). Row i is the state at t[i].
  nfev: int
    Number of right-hand-side evaluations used.
  """

  t: np.ndarray
  y: np.ndarray
  nfev: int

# Integrating with fixed steps
def fixed_step_solve(f, t_span, y0, h, step=rk4_step, args=()):
  """
  Function:
  Integrate dy/dt = f(t, y) for the range of t_span with fixed steps.


  Parameters:
  f: callable
    Right-hand side ``f(t, y, *args)``.
  t_span : tuple of float
    ``(t0, t_end)`` [s], with ``t_end > t0``.
  y0: array_like
    Initial state.
  h: float
    Step size [s], positive.
  step: callable
    One-step method, e.g. :func:`euler_step` or :func:`rk4_step`.
  args: tuple
    Extra arguments passed to ``f``.

  Returns:
  FixedStepResult

  Raises:
  ValueError
    If ``h <= 0`` or ``t_end <= t0``.
  """
  t0, t_end = float(t_span[0]), float(t_span[1])
  if not (h > 0):
    raise ValueError(f"step size h must be positive")
  if not (t_end > t0):
    raise ValueError(f"t_end must be greater than t0")

  # The small tolerance stops 10.000000000001 steps becoming 11 because of round-off.
  n_steps = int(np.ceil((t_end - t0) / h - 1e-9))

  # Pre-allocate output arrays (faster than appending to a list in a loop).
  y0 = np.asarray(y0, dtype=float)
  t_out = np.empty(n_steps + 1)
  y_out = np.empty((n_steps + 1, y0.size))
  t_out[0], y_out[0] = t0, y0

  for i in range(n_steps):
    h_i = min(h, t_end - t_out[i])
    y_out[i + 1] = step(f, t_out[i], y_out[i], h_i, *args)
    t_out[i + 1] = t_out[i] + h_i

  nfev = n_steps * STAGES.get(step, 0)
  return FixedStepResult(t=t_out, y=y_out, nfev=nfev)
