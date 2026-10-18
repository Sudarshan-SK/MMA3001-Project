

const_atm = ConstantAtmosphere()
calm = ConstantWind()

def vertical_rhs(t, y):
    # The full 3D model, fed the simplified problem that has an exact solution.
    return descent_rhs(t, y, MASS, DROGUE.cds, const_atm, calm)

y0 = [0.0, 0.0, APOGEE, 0.0, 0.0, 0.0]     # at rest at apogee
T_END, H_COARSE = 10.0, 0.5

t_fine = np.linspace(0, T_END, 400)
speed_exact, _ = analytic_fall(t_fine, MASS, DROGUE.cds, RHO_TEST)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(t_fine, speed_exact, "k-", lw=2, label="Exact: $v_t \\tanh(gt/v_t)$")
for step, marker in [(euler_step, "o"), (rk4_step, "s")]:
    res = fixed_step_solve(vertical_rhs, (0, T_END), y0, H_COARSE, step=step)
    ax.plot(res.t, -res.y[:, 5], marker, ms=4, label=f"{step.__name__}, h = {H_COARSE} s")
ax.set(xlabel="Time since release [s]", ylabel="Downward speed [m/s]",
       title="Drogue fall from rest: numerical vs exact")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_exact_vs_numerical.png"))
