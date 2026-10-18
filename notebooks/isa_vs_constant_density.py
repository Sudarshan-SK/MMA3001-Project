

isa = ISAAtmosphere()
rho_pad = float(isa.density(0.0))
start = [0.0, 0.0, APOGEE, 0.0, 0.0, 0.0]

rows = []
for label, atm in [("ISA (pad-anchored)", isa), (f"Constant {rho_pad:.3f} kg/m3", ConstantAtmosphere())]:
    cfg = DescentConfig(mass=MASS, drogue=DROGUE, main=MAIN, main_deploy_alt=MAIN_ALT, atmosphere=atm)
    res = simulate_descent(cfg, start)
    rows.append((label, res.t_main, res.t_land, res.nfev))

print(f"{'Model':28s} {'t_main [s]':>10s} {'t_land [s]':>10s} {'nfev':>6s}")
for r in rows:
    print(f"{r[0]:28s} {r[1]:10.1f} {r[2]:10.1f} {r[3]:6d}")
print(f"Difference in descent time: {rows[1][2] - rows[0][2]:+.1f} s "
      f"({100 * (rows[1][2] - rows[0][2]) / rows[0][2]:+.1f} %)")
# The measured Meridian descent took about 206 s (apogee at ~25 s, landing at ~231 s).
