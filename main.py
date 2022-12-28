import numpy as np

from methods import FracStep, StreamFunctionVorticity, PoissonIterative
from utils import (plot_one_contourf, plot_one_contour, plot_one_streamlines, animate)


MAE = lambda array1, array2, domain: np.linalg.norm(array1[domain] - array2[domain])
RANGE = lambda array1, array2, domain: np.max(np.abs(array1[domain] - array2[domain]))

def main(method):
    if method == "Frac_Step":
        ploter = lambda u, v, velocity, pressure, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 0.07), 
                                          plot_one_contourf((np.transpose(pressure)), dx, dy, "pressure at " + str(round((t + 1) * dt, 3)) + "s", "pressure[Pa]", 0.0, 5000.0), 
                                          plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))        

        animator = lambda velocity, pressure, dx, dy: (animate(velocity, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 0.07),
                    animate(pressure, dx, dy, "pressure", "pressure[Pa]", 0, 5000.0))
        method = FracStep("sample_cases/small_case", RANGE, ploter, animator)
        method.solve(100, 10)
    elif method == "SFV":
        ploter = lambda u, v, velocity, w, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 6.0),
                                          plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))

        animator = lambda velocity, w, dx, dy: (animate(velocity, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 6.0), 
                                                plot_one_contour((np.transpose(w[len(w) - 1])), dx, dy, "vorticity at final"))

        method = StreamFunctionVorticity("sample_cases/driven_cavity_case", MAE, ploter, animator)
        method.solve(int(0.2/0.002), 10)
    elif method == "PoissonIterative":
        ploter = lambda X, dx, dy: (plot_one_contourf(X.transpose(), dx, dy, "temperature", "temperature[◦C]", 0.0, 1.0))
        method = PoissonIterative("sample_cases/heat_diffusion_case", MAE, ploter)
        method.solve(0)
        
        
if __name__ == "__main__":
    main("Frac_Step")