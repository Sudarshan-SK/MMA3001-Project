

T_CHECK = 3.0
steps = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
exact_at_check, _ = analytic_fall(T_CHECK, MASS, DROGUE.cds, RHO_TEST)

errors = {}
for step in (euler_step, rk4_step):
    errors[step.__name__] = np.array([
        abs(-fixed_step_solve(vertical_rhs, (0, T_CHECK), y0, h, step=step).y[-1, 5] - exact_at_check)
        for h in steps])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, err in errors.items():
    usable = err > 1e-12                       # below ~1e-12 round-off dominates; exclude from the fit
    order, _ = np.polyfit(np.log(steps[usable]), np.log(err[usable]), 1)
    ax.loglog(steps, err, "o-", label=f"{name}: observed order {order:.2f}")
    print(f"{name:10s} observed order = {order:.3f}")

# Reference slopes, anchored at the largest step so they sit next to the data
for p, style in [(1, "k--"), (4, "k:")]:
    ref = errors["euler_step" if p == 1 else "rk4_step"][0] * (steps / steps[0]) ** p
    ax.loglog(steps, ref, style, lw=1, label=f"slope {p}")
ax.set(xlabel="Step size h [s]", ylabel=f"|speed error| at t = {T_CHECK:g} s [m/s]",
       title="Convergence of the hand-written integrators")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_convergence.png"))
