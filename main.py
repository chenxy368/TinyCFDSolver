import numpy as np

from utils import FracStepSolver
from utils import plot_one_contourf, plot_one_streamlines, animate
from utils import StaggeredGridLoader

MAE = lambda array1, array2, domain: np.linalg.norm(array1[domain] - array2[domain])
RANGE = lambda array1, array2, domain: np.max(np.abs(array1[domain] - array2[domain]))

def main():
    grid_infomation = StaggeredGridLoader("sample_cases/small_case")

    ploter = lambda u, v, velocity, pressure, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 0.07), 
                                          plot_one_contourf((np.transpose(pressure)), dx, dy, "pressure at " + str(round((t + 1) * dt, 3)) + "s", "pressure[Pa]", 0.0, 5000.0), 
                                          plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))        

    animator = lambda velocity, pressure, dx, dy: (animate(velocity, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 0.07),
                    animate(pressure, dx, dy, "pressure", "pressure[Pa]", 0, 5000.0))

    solver = FracStepSolver(0.001, 1000, 0.001, 0.1, 0.1, grid_infomation.u_shape, grid_infomation.v_shape, grid_infomation.p_shape, 
             grid_infomation.u_interior, grid_infomation.v_interior, grid_infomation.p_interior, grid_infomation.u_exterior, 
             grid_infomation.v_exterior, grid_infomation.p_exterior, RANGE, ploter, animator, grid_infomation.u_boundaries, 
             grid_infomation.v_boundaries, grid_infomation.p_boundaries, 0.9, 0)
    solver.solve(100, 3)

if __name__ == "__main__":
    main()