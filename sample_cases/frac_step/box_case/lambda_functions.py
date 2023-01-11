# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:25:00 2023

@author: Xinyang Chen
"""
from utils.plots import plot_one_contourf, plot_one_streamlines, animate
import numpy as np
# Debug ploter
#ploter = lambda u, v, velocity, pressure, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 0.07), 
#                                   plot_one_contourf((np.transpose(pressure)), dx, dy, "pressure at " + str(round((t + 1) * dt, 3)) + "s", "pressure[Pa]", 0.0, 7000.0), 
#                                   plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))        

ploter = None
animator = lambda velocity, pressure, dx, dy: (animate(velocity, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 0.065),
             animate(pressure, dx, dy, "pressure", "pressure[Pa]", 0, 8000.0))
RANGE = lambda array1, array2, domain: np.max(np.abs(array1[domain] - array2[domain]))

lambda_list = [RANGE, ploter, animator]