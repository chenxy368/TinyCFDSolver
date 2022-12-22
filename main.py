import numpy as np

from utils import FracStepSolver
from utils import plot_one_contourf, plot_one_contour, plot_one_streamlines, animate
from utils import StaggeredGridLoader, BlendSchemeGridLoader2D
from utils import StreamFunctionVortex

MAE = lambda array1, array2, domain: np.linalg.norm(array1[domain] - array2[domain])
RANGE = lambda array1, array2, domain: np.max(np.abs(array1[domain] - array2[domain]))

def main(method):
    if method == "Frac_Step":
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
    elif method == "SFV":
        grid_infomation = BlendSchemeGridLoader2D("sample_cases/driven_cavity_case")
        ploter = lambda u, v, velocity, w, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 6.0),
                                          plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))

        animator = lambda velocity, w, dx, dy: (animate(velocity, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 6.0), 
                                                plot_one_contour((np.transpose(w)), dx, dy, "vorticity at final"))

        solver = StreamFunctionVortex(0.01 , 0.01 , 0.002 , 0.01, grid_infomation.shape, grid_infomation.interior, grid_infomation.blend_interior, 
                                      grid_infomation.exterior, MAE, grid_infomation.u_boundaries, grid_infomation.v_boundaries, grid_infomation.psi_boundaries , 
                                      grid_infomation.wu_boundaries, grid_infomation.wv_boundaries, grid_infomation.wx_plus_boundaries, grid_infomation.wy_plus_boundaries, 
                                      grid_infomation.wx_minus_boundaries, grid_infomation.wy_minus_boundaries, ploter, animator, 0.25, 0, 10**-4)
        solver.solve(int(3.0/0.002), 3)
        
if __name__ == "__main__":
    main("SFV")