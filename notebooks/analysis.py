
# File for all of the analysis, imports and plotting


# File for all of the analysis, imports and plotting

# Setting up paths, imports and the plotting style
import os, sys, json, time, cProfile, pstats, io, tracemalloc
try:
  # Done to run in Colab
  from google.colab import drive
  drive.mount("/content/drive")
  REPO = "/content/drive/MyDrive/MMA3001-Project"
  # Cloning the repo instead
  if not os.path.isdir(REPO):
    REPO = "/content/MMA3001-Project"
except ImportError:
  REPO = os.path.abspath("..")
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, "data", "processed")
RAW_GFC = os.path.join(REPO, "data", "raw", "Electronics Bay", "GFC")
FIG = os.path.join(REPO, "reports", "figures")
os.makedirs(FIG, exist_ok=True)
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

# Using a custom colour palette
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "axes.prop_cycle": cycler(color=PALETTE),
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,})
INK, MUTED = "#2b2b2b", "#8a8a85"
def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")

from src.constants import G0
from src.atmosphere import ISAAtmosphere, ConstantAtmosphere
from src.wind import ConstantWind, ProfileWind, wind_components
from src.integrators import euler_step, rk4_step, fixed_step_solve
from src.analytical import analytic_fall, terminal_velocity
from src.descent import Parachute, DescentConfig, descent_rhs, simulate_descent
from src.flightdata import (make_processed_data, load_processed, altitude_from_pressure, latlon_to_en,
                            resample_uniform, descent_rate, detect_launch, detect_apogee, detect_landing)
from src.identify import (cds_quasi_steady, fit_cds, bootstrap_cds, predict_fall_time, bearing_deg,
                          hand_calc_drift, gps_wind_profile, landing_error)
from src import deployment as dep
from src import uncertainty as unc
RESULTS = {}

