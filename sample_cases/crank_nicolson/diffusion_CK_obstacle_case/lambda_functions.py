# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:16:57 2023

@author: Xinyang Chen
"""
from utils import (plot_one_contourf, animate)
#Debug ploter
#ploter = lambda X, dx, dy, dt, t: (plot_one_contourf(X.transpose(), dx, dy, "temperature at " + str(round((t + 1) * dt, 3)) + "s", "temperature[K]", 0.0, 1650.0))
ploter = None
animator = lambda X, dx, dy: (animate(X, dx, dy, "temperature", "temperature[K]", 0.0, 1650.0))

lambda_list = [ploter, animator]