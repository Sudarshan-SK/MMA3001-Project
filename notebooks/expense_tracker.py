

for method in ["RK45", "DOP853", "Radau", "LSODA"]:
    r = simulate_descent(cfg, [0.0, 0.0, APOGEE, 0.0, 0.0, 0.0], method=method, rtol=1e-8, atol=1e-8)
    print(f"{method:7s} nfev = {r.nfev:6d}   landing = ({r.landing_xy[0]:8.2f}, {r.landing_xy[1]:8.2f}) m")
