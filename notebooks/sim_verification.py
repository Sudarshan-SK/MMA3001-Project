
import numpy as np
import matplotlib.pyplot as plt

from src.analytical import analytic_fall, terminal_velocity
from src.atmosphere import ISAAtmosphere, ConstantAtmosphere
from src.descent import Parachute, DescentConfig, descent_rhs, simulate_descent
from src.integrators import euler_step, rk4_step, fixed_step_solve
from src.wind import ConstantWind, wind_components

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


MASS = 20.0                                # kg  - PLACEHOLDER, see table
DROGUE = Parachute("drogue", 0.9144, 1.3)
MAIN = Parachute("main", 2.4384, 1.8)
PAD_P0, PAD_T0 = 99434.0, 300.0            # Pa, K  - T0 is a PLACEHOLDER
MAIN_ALT = 460.0                           # m AGL
APOGEE = 3422.0                            # m AGL
RHO_TEST = 1.1                             # kg/m^3, constant density for the exact-solution checks

print(f"Drogue CdS = {DROGUE.cds:.3f} m^2,  main CdS = {MAIN.cds:.3f} m^2")
print(f"Terminal speed at rho={RHO_TEST}: drogue {terminal_velocity(MASS, DROGUE.cds, RHO_TEST):.2f} m/s, "
      f"main {terminal_velocity(MASS, MAIN.cds, RHO_TEST):.2f} m/s")
