# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:32:15 2023

@author: Xinyang Chen
"""
from utils import (plot_one_contourf, plot_one_contour, plot_one_streamlines, animate)
import numpy as np
# Debug Ploter
#ploter = lambda u, v, velocity, w, dx, dy, dt, t: (plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 6.0),
#                                   plot_one_streamlines(u.transpose(), v.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's'))
ploter = None

animator = lambda velocity, w, dx, dy: (animate(velocity, dx, dy, "speed", "speed[m/s]", 0.0, 6.0), 
                                         plot_one_contour((w[len(w) - 1]), dx, dy, "vorticity at final"))
MAE = lambda array1, array2, domain: np.linalg.norm(array1[domain] - array2[domain])

lambda_list = [MAE, ploter, animator]