# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:11:46 2023

@author: Xinyang Chen
"""

from utils import plot_one_contourf, animate

# Debug ploter
#ploter = lambda X, dx, dy, dt, t: (plot_one_contourf(X.transpose(), dx, dy, "temperature at " + str(round((t + 1) * dt, 3)) + "s", "temperature[$^\circ$C]", 0.0, 1.05))        
ploter = None
animator = lambda X, dx, dy: (animate(X, dx, dy, "temperature", "temperature[$^\circ$C]", 0.0, 1.0))

lambda_list = [ploter, animator]