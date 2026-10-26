
# File for all of the analysis, imports and plotting

# Setting up paths, imports and the plotting style
import os, sys, json, time, cProfile, pstats, io, tracemalloc

sys.path.insert(0, REPO)
DATA = "/content/MMA3001-Project/data/processed"
RAW_GFC = os.path.join(REPO, "data", "raw", "Electronics Bay", "GFC")
FIG = os.path.join(REPO, "reports", "figures")
os.makedirs(FIG, exist_ok=True)


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
from src.drift import (bearing_deg, hand_calc_drift, gps_wind_profile, landing_error)
from src import uncertainty as unc
RESULTS = {}


# Parameters

# These have been gathered from HPR data or using placeholder values

MASS = 33.0
PAD_T0 = 300.0
SURFACE_WIND = None
DROGUE = Parachute("drogue", 0.9144, 1.3)
MAIN = Parachute("main", 2.4384, 1.8)

WINDOWS = {"drogue": (32.0, 163.0), "main": (175.0, 229.0)} # steady phases


# Verification of the descent model
# Numerical vs Exact with a coarse step and h = 0.5s
RHO_TEST = 1.1
const_atm, calm = ConstantAtmosphere(RHO_TEST), ConstantWind()
def vertical_rhs(t, y):
  return descent_rhs(t, y, MASS, DROGUE.cds, const_atm, calm)
y0_test = [0.0, 0.0, 3000.0, 0.0, 0.0, 0.0]

t_fine = np.linspace(0, 10, 400)
v_exact, _ = analytic_fall(t_fine, MASS, DROGUE.cds, RHO_TEST)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(t_fine, v_exact, color=INK, lw=2, label="exact: $v_t\\tanh(gt/v_t)$")
for step, marker in [(euler_step, "o"), (rk4_step, "s")]:
  r = fixed_step_solve(vertical_rhs, (0, 10), y0_test, 0.5, step=step)
  ax.plot(r.t, -r.y[:, 5], marker, ms=4, label=f"{step.__name__}, h = 0.5 s")
ax.set(xlabel="time since release [s]", ylabel="downward speed [m/s]", title="Drogue fall from rest: numerical vs exact")
ax.legend()
save(fig, "drogue_fall_from_rest_num_vs_exact.png")

# Performing a convergence study
# There is an error at t = 3s while the speed is still changing

steps = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
v3, _ = analytic_fall(3.0, MASS, DROGUE.cds, RHO_TEST)
errors, orders = {}, {}
for step in (euler_step, rk4_step):
  errors[step.__name__] = np.array([abs(-fixed_step_solve(vertical_rhs, (0, 3.0), y0_test, h, step=step).y[-1, 5] - v3)
                                    for h in steps])
  # Excluding the round-off floor from the fit
  ok = errors[step.__name__] > 1e-12
  orders[step.__name__] = np.polyfit(np.log(steps[ok]), np.log(errors[step.__name__][ok]), 1)[0]

# Same study measured near terminal velocity
v10, _ = analytic_fall(10.0, MASS, DROGUE.cds, RHO_TEST)
e10 = [abs(-fixed_step_solve(vertical_rhs, (0, 10.0), y0_test, h, step=euler_step).y[-1, 5] - v10) for h in steps[:4]]
orders["euler_at_10s"] = np.polyfit(np.log(steps[:4]), np.log(e10), 1)[0]

fig, ax = plt.subplots(figsize=(6, 3.8))
for name, err in errors.items():
  ax.loglog(steps, err, "o-", label=f"{name}: observed order {orders[name]:.2f}")
for p, style in [(1, "--"), (4, ":")]:
  ref = errors["euler_step" if p == 1 else "rk4_step"][0] * (steps / steps[0]) ** p
  ax.loglog(steps, ref, style, color=MUTED, lw=1, label=f"reference slope {p}")
ax.set(xlabel="step size h [s]", ylabel="|speed error| at t = 3 s [m/s]", title="Convergence of the hand-written integrators")
ax.legend()
save(fig, "drogue_fall_from_rest_convergence.png")
RESULTS["descent_model"] = {"order_euler": orders["euler_step"], "order_rk4": orders["rk4_step"],
                   "order_euler_near_terminal": orders["euler_at_10s"],
                   "rk4_error_h0.00625": float(errors["rk4_step"][-1])}
RESULTS["descent_model"]


# Selecting the model and comparing ISA and constant density
baro, accel, gps, ev = load_processed(DATA)
isa = ISAAtmosphere(p0=ev["pad_pressure_pa"], T0=PAD_T0)
rho_pad = float(isa.density(0.0))
start = [0.0, 0.0, 3583.0, 0.0, 0.0, 0.0]
rows = []
for label, atm_ in [("ISA, pad-anchored", isa), (f"constant {rho_pad:.3f} kg/m3", ConstantAtmosphere(rho_pad))]:
  r = simulate_descent(DescentConfig(MASS, DROGUE, MAIN, 456.0, atm_), start)
  rows.append((label, r.t_main, r.t_land, r.nfev))
isa_table = pd.DataFrame(rows, columns=["atmosphere", "t_main [s]", "t_land [s]", "nfev"])
RESULTS["descent_model"]["isa_vs_constant_descent_time_diff_s"] = float(rows[1][2] - rows[0][2])

solver_rows = []
for method in ["RK45", "DOP853", "Radau", "BDF", "LSODA"]:
  t0 = time.perf_counter()
  r = simulate_descent(DescentConfig(MASS, DROGUE, MAIN, 456.0, isa), start, method=method, rtol=1e-8, atol=1e-8)
  solver_rows.append((method, r.nfev, time.perf_counter() - t0, r.t_land))
descent_solvers = pd.DataFrame(solver_rows, columns=["method", "nfev", "time [s]", "t_land [s]"])
RESULTS["descent_model"]["descent_solver_nfev"] = dict(zip(descent_solvers.method, descent_solvers.nfev))
display(isa_table.round(2)); descent_solvers.round(4)





# Flight data analysis from onboard flight computers
# Organising the data and events
atm = ISAAtmosphere(p0=ev["pad_pressure_pa"], T0=PAD_T0)
baro["alt"] = altitude_from_pressure(baro.pressure, ev["pad_pressure_pa"], PAD_T0)
gps["alt"] = gps.alt_gps - ev["pad_gps_alt_m"]
a2 = accel[accel.sensor == "ACCEL2"]
a2_mag = np.sqrt(a2.ax**2 + a2.ay**2 + a2.az**2).to_numpy()
t_launch = detect_launch(a2.t.to_numpy(), a2_mag)
t_apo, h_apo = detect_apogee(baro.t, baro.alt)
t_land = detect_landing(baro.t, baro.alt, t_after=t_apo)
events_table = pd.DataFrame({"detected [s]": [t_launch, t_apo, ev["logged"]["main"], t_land],
                             "GFC logged [s]": [0.0, ev["logged"]["drogue"], ev["logged"]["main"], ev["logged"]["landing"]]},
                            index=["launch", "apogee", "main release", "landing"])
RESULTS["flight_data"] = {"t_apogee": t_apo, "h_apogee": h_apo, "t_land": t_land, "gfc_logged": ev["logged"]}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
ax1.plot(baro.t, baro.alt, label=f"baro, from pressure (T0 = {PAD_T0:.0f} K)")
ax1.plot(gps.t, gps.alt, ".", ms=3, label="GPS altitude")
ax1.set(ylabel="altitude AGL [m]", title="Meridian, Flight Data")
ax2.plot(a2.t, a2_mag / G0, lw=0.6, label="ASM330 |a| (~73 Hz)")
ax2.set(xlabel="time since launch [s]", ylabel="specific force [g]", ylim=(-0.3, 15))
for ax in (ax1, ax2):
  for t_e in (t_apo, ev["logged"]["main"], t_land):
    ax.axvline(t_e, color=MUTED, ls=":", lw=1)
  for lo, hi in WINDOWS.values():
    ax.axvspan(lo, hi, color=PALETTE[2], alpha=0.07, lw=0)
    ax.legend(loc="upper right")
save(fig, "flight_overview.png")
events_table.round(2)


# Comparing altitude results based on the source
gps_apogee = float(gps[(gps.t > 20) & (gps.t < 35)].alt.max())
rows = [("device-logged", float(baro.alt_agl_device.max()))]
for T0 in [288.15, 295.0, 300.0, 305.0, 310.0]:
  rows.append((f"from pressure, T0 = {T0:.0f} K", float(altitude_from_pressure(baro.pressure, ev["pad_pressure_pa"], T0).max())))
rows.append(("GPS (independent of temperature)", gps_apogee))
alt_table = pd.DataFrame(rows, columns=["altitude source", "apogee [m AGL]"])
RESULTS["flight_data"].update({"apogee_device": rows[0][1], "apogee_gps": gps_apogee})
alt_table.round(0)


# Descent-rate method selection
noise_std = float(baro[baro.t < -1].alt.std())
rng = np.random.default_rng(0)
t_syn = np.arange(0, 120, 1 / 14.5)
h_syn = 3000 - 20 * t_syn + rng.normal(0, noise_std, t_syn.size)
rows = [("central", "-", np.sqrt(np.nanmean((descent_rate(t_syn, h_syn, "central")[1] + 20) ** 2)))]
for method in ["savgol", "linefit"]:
  for w in [0.5, 1, 2, 4, 8]:
    rows.append((method, w, np.sqrt(np.nanmean((descent_rate(t_syn, h_syn, method, window_s=w)[1] + 20) ** 2))))
rate_table = pd.DataFrame(rows, columns=["method", "window [s]", "RMSE [m/s]"])
METHOD, WINDOW = "linefit", 2.0
t_v, v = descent_rate(baro.t, baro.alt, METHOD, window_s=WINDOW)
h_v = np.interp(t_v, baro.t, baro.alt)
in_phase = {ph: (t_v > lo) & (t_v < hi) for ph, (lo, hi) in WINDOWS.items()}
v_stage = {ph: float(-np.nanmedian(v[m])) for ph, m in in_phase.items()}
RESULTS["flight_data"].update({"baro_noise_m": noise_std, "v_drogue": v_stage["drogue"], "v_main": v_stage["main"]})
print(f"pad noise {noise_std:.2f} m; chosen {METHOD}, {WINDOW} s window; descent rates {v_stage}")
rate_table.round(3)





# Comparing the Landing drift between the baseline and the model
east, north = latlon_to_en(gps.lat, gps.lon, ev["pad_lat"], ev["pad_lon"])
gps["east"], gps["north"] = east, north
gps["alt_baro"] = np.interp(gps.t, baro.t, baro.alt)
apo = gps.iloc[(gps.t - t_apo).abs().argmin()]
landing = gps[gps.t > t_land + 5][["east", "north"]].mean().to_numpy()
measured_drift = landing - np.array([apo.east, apo.north])
descent_fixes = gps[(gps.t > t_apo + 3) & (gps.t < t_land)]
gps_wind = gps_wind_profile(descent_fixes.t, descent_fixes.east, descent_fixes.north, descent_fixes.alt_baro, bin_m=250.0)
main_alt = float(np.interp(ev["logged"]["main"], baro.t, baro.alt))
y0_apo = [0.0, 0.0, h_apo, 0.0, 0.0, 0.0]

cases, tracks = {}, {}
if SURFACE_WIND is not None:
  surface_uv, label_a = wind_components(*SURFACE_WIND), "A: hand calc, surface wind"
else:
  surface_uv, label_a = np.array([gps_wind.u_east[0], gps_wind.v_north[0]]), "A*: hand calc, GPS stand-in wind"
cases[label_a] = hand_calc_drift(h_apo, main_alt, v_stage["drogue"], v_stage["main"], surface_uv)[0]
for label, (d_, m_) in {"E: ODE manufacturer CdS, GPS wind": (DROGUE, MAIN)}.items():
  r = simulate_descent(DescentConfig(MASS, d_, m_, main_alt, atm, gps_wind), y0_apo)
  cases[label], tracks[label] = r.landing_xy, r.y[:, :2]
rows = []
for label, xy in cases.items():
  e = landing_error(xy, measured_drift)
  rows.append({"case": label, "east [m]": xy[0], "north [m]": xy[1], "miss [m]": e["distance"],
                 "miss [% of drift]": 100 * e["distance"] / np.hypot(*measured_drift),
                 "bearing error [deg]": e["bearing_error"], "range error [m]": e["range_error"]})
drift_table = pd.DataFrame(rows)
RESULTS["flight_data"]["drift"] = {"measured_m": float(np.hypot(*measured_drift)), "measured_bearing": bearing_deg(*measured_drift),
                            "cases": {r["case"]: {"miss_m": r["miss [m]"], "miss_pct": r["miss [% of drift]"]} for r in rows},
                            "wind_low_ms": float(np.hypot(gps_wind.u_east[0], gps_wind.v_north[0])),
                            "wind_high_ms": float(np.hypot(gps_wind.u_east[-1], gps_wind.v_north[-1]))}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2.4]})
ax1.plot(np.hypot(gps_wind.u_east, gps_wind.v_north), gps_wind.heights, "o-")
ax1.set(xlabel="wind speed [m/s]", ylabel="altitude AGL [m]", title="Wind from GPS drift")
ax2.plot(descent_fixes.east - apo.east, descent_fixes.north - apo.north, ".", color=INK, ms=3, label="GPS track")
for i, (label, tr) in enumerate(tracks.items()):
    ax2.plot(tr[:, 0], tr[:, 1], color=PALETTE[i], label=label)
for i, (label, xy) in enumerate(cases.items()):
    ax2.plot(*xy, "x", ms=8, mew=2, color=PALETTE[(i + 2) % 8] if i == 0 else PALETTE[i - 1])
ax2.plot(0, 0, "^", color=INK, ms=7, label="apogee point")
ax2.plot(*measured_drift, "*", color=INK, ms=12, label="measured landing")
ax2.annotate(label_a.split(":")[0], cases[label_a], textcoords="offset points", xytext=(6, 4), color=INK)
ax2.set(xlabel="east of apogee point [m]", ylabel="north of apogee point [m]", title="Descent ground track", aspect="equal")
ax2.legend(loc="upper left", fontsize=7)
save(fig, "ground_track.png")
drift_table.round(1)




# Creating the Monte Carlo landing dispersion and analysis
SPEC = unc.UncertaintySpec(mass=MASS, cds_drogue=DROGUE.cds, cds_main=MAIN.cds,
                           main_alt=main_alt, apogee_alt=h_apo)

# Comparing accuracy and speed
s50 = unc.sample_inputs(SPEC, 50, seed=1)
t_loop, loop50 = unc.best_time(lambda: unc.simulate_descent_loop(s50, atm, gps_wind, rtol=1e-8, atol=1e-8), repeats=1)
t_batch, batch50 = unc.best_time(lambda: unc.simulate_descent_batch(s50, atm, gps_wind, h=0.1), repeats=3)
miss = np.hypot(*(batch50["landing_xy"] - loop50["landing_xy"]).T)
step_rows = []
for h in [0.8, 0.4, 0.2, 0.1, 0.05, 0.025]:
  b = unc.simulate_descent_batch(s50, atm, gps_wind, h=h)
  step_rows.append((h, float(np.hypot(*(b["landing_xy"] - loop50["landing_xy"]).T).max())))
step_table = pd.DataFrame(step_rows, columns=["h [s]", "max landing difference vs solve_ivp [m]"])
order_batch = float(np.polyfit(np.log(step_table.iloc[:, 0]), np.log(step_table.iloc[:, 1]), 1)[0])
scaling = []
for n in [10, 100, 1000, 5000]:
  s_n = unc.sample_inputs(SPEC, n, seed=2)
  tb, _ = unc.best_time(lambda: unc.simulate_descent_batch(s_n, atm, gps_wind, h=0.1), repeats=1)
  scaling.append((n, tb, t_loop / 50 * n))
scaling = pd.DataFrame(scaling, columns=["N", "vectorised [s]", "loop (extrapolated) [s]"])
RESULTS["monte_carlo"] = {"loop_s_per_sample": t_loop / 50, "batch_s_50": t_batch, "speedup_50": t_loop / t_batch,
                   "max_miss_vs_solve_ivp_m": float(miss.max()), "observed_order_batch": order_batch,
                   "scaling": scaling.to_dict("records")}
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.loglog(scaling.N, scaling["loop (extrapolated) [s]"], "o--", label="solve_ivp loop")
ax.loglog(scaling.N, scaling["vectorised [s]"], "o-", label="vectorised RK4")
ax.set(xlabel="Monte Carlo samples N", ylabel="wall time [s]", title="Monte Carlo cost scaling")
ax.legend()
save(fig, "mc_cost_scaling.png")
print(f"speed-up at N=50: {t_loop / t_batch:.0f}x; max landing difference {miss.max():.2f} m; batch observed order {order_batch:.2f}")
display(step_table.round(3)); scaling.round(3)



# Landing ellipse and checking for Monte Carlo convergence
N_MC = 5000
mc = unc.simulate_descent_batch(unc.sample_inputs(SPEC, N_MC, seed=42), atm, gps_wind, h=0.1)
ell = unc.landing_ellipse(mc["landing_xy"])
inside = unc.inside_ellipse(measured_drift, ell)
Ns = [30, 100, 300, 1000, 3000]
seeds = range(10)
conv = np.array([[unc.landing_ellipse(unc.simulate_descent_batch(unc.sample_inputs(SPEC, n, seed=100 + k), atm, gps_wind,
                                                                    h=0.1)["landing_xy"])["semi_major"] for k in seeds] for n in Ns])
spread = conv.std(axis=1) / conv.mean(axis=1)
conv_slope = float(np.polyfit(np.log(Ns), np.log(spread), 1)[0])
n_needed = int(np.ceil(Ns[0] * (spread[0] / 0.02) ** 2))     # from the 1/sqrt(N) law, for 2 % spread
RESULTS["monte_carlo"].update({"N_mc": N_MC, "ellipse_semi_major": ell["semi_major"], "ellipse_semi_minor": ell["semi_minor"],
                        "ellipse_bearing_major": ell["bearing_major_deg"], "ellipse_centre": ell["centre"].tolist(),
                        "measured_inside": inside, "convergence_slope": conv_slope, "n_for_2pct": n_needed,
                        "spread_at_N": dict(zip(map(str, Ns), spread.tolist()))})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.6, 1]})
ax1.plot(*mc["landing_xy"].T, ".", ms=1.2, alpha=0.35, color=PALETTE[0], label=f"{N_MC} Monte Carlo landings")
ax1.plot(*unc.ellipse_outline(ell).T, color=PALETTE[1], lw=2, label="95 % ellipse")
ax1.plot(*measured_drift, "*", color=INK, ms=13, label="measured landing")
ax1.plot(0, 0, "^", color=INK, ms=7, label="apogee point")
ax1.set(xlabel="east of apogee point [m]", ylabel="north of apogee point [m]", aspect="equal", title="Landing dispersion")
ax1.legend(loc="upper left", fontsize=7)
ax2.loglog(Ns, spread, "o-", label=f"observed (slope {conv_slope:.2f})")
ax2.loglog(Ns, spread[0] * (np.array(Ns) / Ns[0]) ** -0.5, "--", color=MUTED, label="1/√N")
ax2.axhline(0.02, color=MUTED, ls=":", lw=1)
ax2.set(xlabel="samples N", ylabel="relative spread of semi-major axis", title="Monte Carlo convergence")
ax2.legend()
save(fig, "monte_carlo_ellipse.png")
print(f"95 % ellipse {ell['semi_major']:.0f} x {ell['semi_minor']:.0f} m, major axis {ell['bearing_major_deg']:.0f} deg; "
      f"measured inside: {inside}; slope {conv_slope:.2f}; N for 2 % spread ≈ {n_needed}")




# Profiling, FLOPs and memory requirements
s1000 = unc.sample_inputs(SPEC, 1000, seed=5)
def profile(fn, sort, n=8):
  prof = cProfile.Profile()
  out = prof.runcall(fn)
  buf = io.StringIO(); pstats.Stats(prof, stream=buf).sort_stats(sort).print_stats(n)
  return out, buf.getvalue()
_, txt = profile(lambda: unc.simulate_descent_batch(s1000, atm, gps_wind, h=0.1, optimised=False), "tottime")
print("ORIGINAL vectorised version, N = 1000, sorted by own time:")
print(txt[txt.find("ncalls"):][:1500])
out, txt = profile(lambda: unc.simulate_descent_batch(s1000, atm, gps_wind, h=0.1), "tottime")
print("OPTIMISED version:")
print(txt[txt.find("ncalls"):][:1200])
_, txt = profile(lambda: unc.simulate_descent_loop(unc.sample_inputs(SPEC, 10, seed=5), atm, gps_wind), "tottime")
print("LOOP baseline, N = 10:")
print(txt[txt.find("ncalls"):][:1200])

t_orig, _ = unc.best_time(lambda: unc.simulate_descent_batch(s1000, atm, gps_wind, h=0.1, optimised=False), repeats=3)
t_1000, _ = unc.best_time(lambda: unc.simulate_descent_batch(s1000, atm, gps_wind, h=0.1), repeats=3)
achieved = out["flops"] / t_1000
tracemalloc.start(); unc.simulate_descent_batch(s1000, atm, gps_wind, h=0.1); peak_batch = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
bytes_per_rhs_per_sample = 20 * 6 * 8
RESULTS["monte_carlo"].update({"flops_estimate_1000": out["flops"], "time_1000_s": t_1000, "time_1000_original_s": t_orig,
                        "achieved_flops": achieved, "peak_memory_batch_1000_MB": peak_batch / 1e6,
                        "arithmetic_intensity_flop_per_byte": unc.FLOPS_PER_RHS / bytes_per_rhs_per_sample})
print(f"N=1000: original {t_orig:.2f} s -> optimised {t_1000:.2f} s ({100 * (1 - t_1000 / t_orig):.0f} % faster); "
      f"{out['flops']:.2e} FLOPs -> {achieved / 1e6:.0f} MFLOP/s; peak memory {peak_batch / 1e6:.1f} MB; "
      f"arithmetic intensity ~{unc.FLOPS_PER_RHS / bytes_per_rhs_per_sample:.2f} FLOP/byte")
