import numpy as np

from methods import FracStepSolver, StreamFunctionVortex
from utils import (plot_one_contourf, plot_one_contour, plot_one_streamlines, animate,
                StaggeredGridLoader, BlendSchemeGridLoader2D)


MAE = lambda array1, array2, domain: np.linalg.norm(array1[domain] - array2[domain])
RANGE = lambda array1, array2, domain: np.max(np.abs(array1[domain] - array2[domain]))

def main(method):
    if method == "Frac_Step":
        grid_infomation = StaggeredGridLoader("sample_cases/box_column_case")

        ploter = lambda u, v, velocity, pressure, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 0.07), 
                                          plot_one_contourf((np.transpose(pressure)), dx, dy, "pressure at " + str(round((t + 1) * dt, 3)) + "s", "pressure[Pa]", 0.0, 5000.0), 
                                          plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))        

        animator = lambda velocity, pressure, dx, dy: (animate(velocity, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 0.07),
                    animate(pressure, dx, dy, "pressure", "pressure[Pa]", 0, 5000.0))
        
        method_info, mesh_data = grid_infomation.load_grid()

        solver = FracStepSolver(method_info, mesh_data, RANGE, ploter, animator)
        solver.solve(100, 10)
    elif method == "SFV":
        grid_infomation = BlendSchemeGridLoader2D("sample_cases/driven_cavity_case")
        ploter = lambda u, v, velocity, w, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 6.0),
                                          plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))

        animator = lambda velocity, w, dx, dy: (animate(velocity, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 6.0), 
                                                plot_one_contour((np.transpose(w)), dx, dy, "vorticity at final"))

        method_info, mesh_data = grid_infomation.load_grid()
        solver = StreamFunctionVortex(method_info, mesh_data, MAE, ploter, animator)
        solver.solve(int(3.0/0.002), 10)
        
        
if __name__ == "__main__":
    solver = main("SFV")