
"""
Using a Monte Carlo for landing dispersion, sensitivity analysis and performance tools.

"""

import time
from dataclasses import dataclass

import numpy as np

from src.constants import G0
from src.descent import DescentConfig, Parachute, simulate_descent
from src.wind import ConstantWind, PowerLawWind, ProfileWind, wind_components

#: chi-squared quantile with 2 degrees of freedom for a 95 % ellipse.
CHI2_2DOF_95 = 5.991464547107979

#: Floating-point operations in one vectorised descent right-hand-side evaluation, per sample
FLOPS_PER_RHS = 50


@dataclass(frozen=True)
class UncertaintySpec:
  """
  Class for the nominal values and 1-sigma uncertainties for the landing Monte Carlo.

  Parameters:
  mass: float
    Descending mass [kg].
  cds_drogue, cds_main: float
    Drag areas [m^2].
  main_alt: float
    Main deployment altitude above the pad [m].
  apogee_alt: float
    Starting altitude [m].
  sigma_mass_frac: float
    Mass uncertainty as a fraction of mass.
  sigma_cds_drogue, sigma_cds_main: float
    Drag-area uncertainties [m^2].
  sigma_wind_scale: float
    Fractional uncertainty in wind speed.
  sigma_wind_dir_deg: float
    Uncertainty in wind direction [deg].
  sigma_main_alt: float
    Main deployment altitude uncertainty [m].
  sigma_apogee_xy: float
    Apogee horizontal position uncertainty [m].
  """

  mass: float
  cds_drogue: float
  cds_main: float
  main_alt: float
  apogee_alt: float
  sigma_mass_frac: float = 0.03
  sigma_cds_drogue: float = 0.0
  sigma_cds_main: float = 0.0
  sigma_wind_scale: float = 0.15
  sigma_wind_dir_deg: float = 10.0
  sigma_main_alt: float = 15.0
  sigma_apogee_xy: float = 3.0

  def __post_init__(self):
    # Require positive values
    for name in ("mass", "cds_drogue", "cds_main", "apogee_alt"):
      if not (getattr(self, name) > 0):
        raise ValueError(f"{name} must be positive")
      # Mains deployment between pad and below apogee
    if not (0 <= self.main_alt < self.apogee_alt):
      raise ValueError("need 0 <= main_alt < apogee_alt")
    # Rejecting negative uncertainty values
    for name, value in self.__dict__.items():
      if name.startswith("sigma") and not (value >= 0):
        raise ValueError(f"{name} must be >= 0")



def sample_inputs(spec, n, seed=0):
  """
  Function to draw Monte Carlo samples.

  Returns:
  dict of numpy.ndarray: (contains the following)
    mass
    cds_drogue
    cds_main
    wind_scale
    wind_rot_deg
    main_alt
    x0
    y0
    apogee_alt
  """
  if n < 1:
    raise ValueError("n must be >= 1")

  # Generating a seed for reproducability
  rng = np.random.default_rng(seed)
  normal = lambda mean, sd: mean + sd * rng.standard_normal(n)
  return {
      "mass": np.clip(normal(spec.mass, spec.sigma_mass_frac * spec.mass), 1e-3, None),
      "cds_drogue": np.clip(normal(spec.cds_drogue, spec.sigma_cds_drogue), 1e-3, None),
      "cds_main": np.clip(normal(spec.cds_main, spec.sigma_cds_main), 1e-3, None),
      "wind_scale": np.clip(normal(1.0, spec.sigma_wind_scale), 0.0, None),
      "wind_rot_deg": normal(0.0, spec.sigma_wind_dir_deg),
      "main_alt": np.clip(normal(spec.main_alt, spec.sigma_main_alt), 0.0, spec.apogee_alt - 1),
      "x0": normal(0.0, spec.sigma_apogee_xy),
      "y0": normal(0.0, spec.sigma_apogee_xy),
      "apogee_alt": spec.apogee_alt,}


def nominal_inputs(spec):
  """The single nominal sample."""
  one = lambda v: np.array([float(v)])
  return {"mass": one(spec.mass), "cds_drogue": one(spec.cds_drogue), "cds_main": one(spec.cds_main),
          "wind_scale": one(1.0), "wind_rot_deg": one(0.0), "main_alt": one(spec.main_alt),
          "x0": one(0.0), "y0": one(0.0), "apogee_alt": spec.apogee_alt}



def wind_uv_batch(wind, z):
  """
  Wind components for multiple altitudes.

  Parameters:
  wind: ConstantWind, PowerLawWind or ProfileWind
  z: numpy.ndarray
    Altitudes [m], shape (N,).

  Returns:
  u, v: numpy.ndarray
    East and north components [m/s], shape (N,).
  """
  z = np.asarray(z, dtype=float)
  if isinstance(wind, ProfileWind):
    return np.interp(z, wind.heights, wind.u_east), np.interp(z, wind.heights, wind.v_north)
  if isinstance(wind, ConstantWind):
    return np.full_like(z, wind.u_east), np.full_like(z, wind.v_north)
  if isinstance(wind, PowerLawWind):
    speed = wind.speed_ref * (np.maximum(z, wind.h_ref) / wind.h_ref) ** wind.alpha
    u_unit, v_unit = wind_components(1.0, wind.direction_from_deg)
    return speed * u_unit, speed * v_unit


@dataclass(frozen=True)
class ScaledRotatedWind:
  """
  Class for a wind profile multiplied by scale and rotated by rot_deg.
  """

  base: object
  scale: float = 1.0
  rot_deg: float = 0.0

  def velocity(self, h):
    u, v = wind_uv_batch(self.base, np.array([h]))
    return np.array(_rotate_scale(u, v, self.scale, self.rot_deg)).ravel()


def _rotate_scale(u, v, scale, rot_deg):
  """Rotate clockwise by rot_deg and multiply by scale."""
  a = np.radians(rot_deg)
  c, s = np.cos(a), np.sin(a)
  return scale * (c * u + s * v), scale * (-s * u + c * v)


def _batch_rhs(Y, mass, cds_d, cds_m, main_alt, atmosphere, wind, scale, rot, g, optimised=True):
  """
  Function for the dY/dt for N descents at once.

  """
  z = Y[:, 2]
  z_air = np.maximum(z, 0.0)
  u, v = wind_uv_batch(wind, z_air)
  if optimised:
    sc, ss = rot
    u, v = sc * u + ss * v, -ss * u + sc * v
  else:
    u, v = _rotate_scale(u, v, scale, rot)
  rel = Y[:, 3:6] - np.column_stack([u, v, np.zeros_like(u)])
  speed = np.sqrt(np.einsum("ij,ij->i", rel, rel))
  cds = np.where(z > main_alt, cds_d, cds_m)
  k = -0.5 * atmosphere.density(z_air) * cds * speed / mass
  acc = rel * k[:, None]
  acc[:, 2] -= g
  return np.column_stack([Y[:, 3:6], acc])


def simulate_descent_batch(samples, atmosphere, wind, h=0.1, t_max=600.0, g=G0, optimised=True):
  """
  Function to integrate all Monte Carlo descents together with fixed-step RK4.

  Parameters:
  samples: dict
    From :func:`sample_inputs` or :func:`nominal_inputs`.
  atmosphere: object with array-capable ``density(h)``
  wind: ConstantWind, PowerLawWind or ProfileWind
  h: float
    Step size [s]. See the notebook for the step-size convergence check.
  t_max: float
    Stop if not every sample has landed by then [s].

  Returns:
  dict (contains the following)
    landing_xy
    t_land
    n_steps
    nfev
    flops

  """
  if not (h > 0):
    raise ValueError("h must be positive")
  atmosphere.density(samples["apogee_alt"])
  n = len(samples["mass"])
  Y = np.zeros((n, 6))
  Y[:, 0], Y[:, 1], Y[:, 2] = samples["x0"], samples["y0"], samples["apogee_alt"]
  if optimised:
    a_rad = np.radians(samples["wind_rot_deg"])
    rot = (samples["wind_scale"] * np.cos(a_rad), samples["wind_scale"] * np.sin(a_rad))
  else:
    rot = samples["wind_rot_deg"]
  args = (samples["mass"], samples["cds_drogue"], samples["cds_main"], samples["main_alt"],
          atmosphere, wind, samples["wind_scale"], rot, g, optimised)
  landed = np.zeros(n, dtype=bool)
  landing_xy = np.full((n, 2), np.nan)
  t_land = np.full(n, np.nan)
  t, steps = 0.0, 0
  while not landed.all() and t < t_max:
    active = ~landed
    Ya = Y[active]
    a = tuple(x[active] if isinstance(x, np.ndarray)
              else (tuple(c[active] for c in x) if isinstance(x, tuple) else x) for x in args)
    k1 = _batch_rhs(Ya, *a)
    k2 = _batch_rhs(Ya + 0.5 * h * k1, *a)
    k3 = _batch_rhs(Ya + 0.5 * h * k2, *a)
    k4 = _batch_rhs(Ya + h * k3, *a)
    Y_new = Ya + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Samples that crossed the ground in this step: interpolate the crossing.
    crossed = Y_new[:, 2] <= 0.0
    if crossed.any():
      frac = Ya[crossed, 2] / (Ya[crossed, 2] - Y_new[crossed, 2])
      idx = np.nonzero(active)[0][crossed]
      landing_xy[idx] = Ya[crossed, :2] + frac[:, None] * (Y_new[crossed, :2] - Ya[crossed, :2])
      t_land[idx] = t + frac * h
      landed[idx] = True
    Y[active] = Y_new
    t += h
    steps += 1
  nfev = 4 * steps * n
  return {"landing_xy": landing_xy, "t_land": t_land, "n_steps": steps, "nfev": nfev,
          "flops": nfev * FLOPS_PER_RHS}


def simulate_descent_loop(samples, atmosphere, wind, method="RK45", rtol=1e-6, atol=1e-6, g=G0):
  """
  Function to simulate the baseline Monte Carlo

  Returns:
  dict (contains the following)
    landing_xy
    t_land
    nfev
  """
  n = len(samples["mass"])
  landing_xy, t_land, nfev = np.empty((n, 2)), np.empty(n), 0
  for i in range(n):
    wind_i = ScaledRotatedWind(wind, samples["wind_scale"][i], samples["wind_rot_deg"][i])
    cfg = DescentConfig(mass=samples["mass"][i],
                        drogue=Parachute("drogue", 1.0, samples["cds_drogue"][i] / (np.pi / 4)),
                        main=Parachute("main", 1.0, samples["cds_main"][i] / (np.pi / 4)),
                        main_deploy_alt=samples["main_alt"][i], atmosphere=atmosphere,
                        wind=wind_i, g=g)
    res = simulate_descent(cfg, [samples["x0"][i], samples["y0"][i], samples["apogee_alt"],
                                  0.0, 0.0, 0.0], method=method, rtol=rtol, atol=atol)
    landing_xy[i], t_land[i], nfev = res.landing_xy, res.t_land, nfev + res.nfev
  return {"landing_xy": landing_xy, "t_land": t_land, "nfev": nfev}



def landing_ellipse(points, chi2=CHI2_2DOF_95):
  """
  Creating a confidence ellipse of 2D landing points.

  Parameters:
  points: array_like
    Landing points (N, 2) [east, north] in metres, N >= 3.
  chi2: float
    Chi-squared quantile (2 dof) for the confidence level.

  Returns:
  dict (contains the following)
    centre
    semi_major
    semi_minor
    bearing_major_deg
    cov

  """
  points = np.asarray(points, dtype=float)

  if points.ndim != 2 or points.shape[1] != 2:
    raise ValueError("points must have shape (n, 2)")

  if points.shape[0] < 3:
    raise ValueError("at least 3 points are required")

  if not np.all(np.isfinite(points)):
    raise ValueError("points must contain only finite values")

  if np.linalg.matrix_rank(np.cov(points, rowvar=False)) < 2:
    raise ValueError("points must have non-degenerate 2D spread")
    
  p = np.asarray(points, dtype=float)
  cov = np.cov(p.T)
  vals, vecs = np.linalg.eigh(cov)
  major = vecs[:, 1]
  return {"centre": p.mean(axis=0), "semi_major": float(np.sqrt(chi2 * vals[1])),
          "semi_minor": float(np.sqrt(chi2 * vals[0])),
          "bearing_major_deg": float(np.degrees(np.arctan2(major[0], major[1])) % 180.0),
          "cov": cov}


def ellipse_outline(ellipse, n=200):
  """Points on the boundary of an ellipse."""
  vals, vecs = np.linalg.eigh(ellipse["cov"])
  theta = np.linspace(0, 2 * np.pi, n)
  circle = np.stack([np.cos(theta), np.sin(theta)])
  scale = np.sqrt(CHI2_2DOF_95 * vals)
  return (vecs @ (scale[:, None] * circle)).T + ellipse["centre"]


def inside_ellipse(point, ellipse, chi2=CHI2_2DOF_95):
  """
  Checking if the point lies inside the ellipse."""
  d = np.asarray(point, dtype=float) - ellipse["centre"]
  return bool(d @ np.linalg.solve(ellipse["cov"], d) <= chi2)


def tornado(model, nominal, deltas):
  """
  Creating a one-at-a-time sensitivity.

  Parameters:
  model: callable
  nominal: dict
    Nominal parameter values.
  deltas: dict
    For each parameter to test, the step.

  Returns:
  list of dict (contains the following)
    parameter
    low
    high
    swing
    nominal_output
  """
  base = model(**nominal)
  rows = []
  for name, d in deltas.items():
    lo = model(**{**nominal, name: nominal[name] - d})
    hi = model(**{**nominal, name: nominal[name] + d})
    rows.append({"parameter": name, "low": lo, "high": hi, "swing": abs(hi - lo),
                  "nominal_output": base})
  return sorted(rows, key=lambda r: r["swing"], reverse=True)


def best_time(fn, repeats=3):
  best, out = np.inf, None
  for _ in range(repeats):
      t0 = time.perf_counter()
      out = fn()
      best = min(best, time.perf_counter() - t0)
  return best, out
