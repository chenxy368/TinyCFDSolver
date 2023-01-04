# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:12:39 2022

@author: Xinyang Chen
"""
import numpy as np

from methods import FracStep, StreamFunctionVorticity, PoissonIterative, UpwindCentral2D, CrankNicolson2D, ADI
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
        method = FracStep("sample_cases/origin_case", RANGE, ploter, animator)
        res = method.solve(100, 10)
    elif method == "SFV":
        ploter = lambda u, v, velocity, w, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 6.0),
                                          plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))

        animator = lambda velocity, w, dx, dy: (animate(velocity, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 6.0), 
                                                plot_one_contour((w[len(w) - 1]), dx, dy, "vorticity at final"))

        method = StreamFunctionVorticity("sample_cases/driven_cavity_case", MAE, ploter, animator)
        res = method.solve(int(0.2/0.002), 10)
    elif method == "PoissonIterative":
        ploter = lambda X, dx, dy: (plot_one_contourf(X.transpose(), dx, dy, "temperature", "temperature[◦C]", 0.0, 1.0))
        method = PoissonIterative("sample_cases/heat_diffusion_case", MAE, ploter)
        res = method.solve(0)
    elif method == "UpwindCentral2D":
        init_path = "sample_cases/advection_diffusion_init_case/init.npy"
        init = np.load(init_path)
        ploter = lambda X, dx, dy, dt, t: (plot_one_contourf(X.transpose(), dx, dy, "temperature at " + str(round((t + 1) * dt, 3)) + "s", "temperature[$^\circ$C]", 0.0, 1.05))        

        animator = lambda X, dx, dy: (plot_one_contourf(X[len(X) - 1], dx, dy, "temperature", "temperature[$^\circ$C]", 0.0, 1.05))
        method = UpwindCentral2D("sample_cases/advection_diffusion_init_case", None, animator, init)
        res = method.solve(int(4/0.002), 100)
    
        #np.save("sample_cases/advection_diffusion_init_case/res.npy", res)
    elif method == "CrankNicolson2D":
        init_path = "sample_cases/diffusion_CK_case/init.npy"
        init = np.load(init_path)

        ploter = lambda X, dx, dy, dt, t: (plot_one_contourf(X.transpose(), dx, dy, "temperature at " + str(round((t + 1) * dt, 3)) + "s", "temperature[K]", 0.0, 1650.0))

        animator = lambda X, dx, dy: (animate(X, dx, dy, "temperature", "temperature[K]", 0.0, 1650.0))
        method = CrankNicolson2D("sample_cases/diffusion_CK_case", ploter, animator, init)

        res =  method.solve(int(0.05/0.001), 3)
        #np.save("sample_cases/diffusion_CK_case/res.npy", res)
    elif method == "ADI":
        ploter = lambda X, dx, dy, dt, t: (plot_one_contourf(X.transpose(), dx, dy, "temperature at " + str(round((t + 1) * dt, 3)) + "s", "temperature[K]", 0.0, 1650.0))

        animator = lambda X, dx, dy: (animate(X, dx, dy, "temperature", "temperature[K]", 0.0, 1650.0))
        method = ADI("sample_cases/diffusion_ADI_case", ploter, animator)
        res = method.solve(int(0.05/0.001), 3)
        
if __name__ == "__main__":
    X = main("PoissonIterative")