# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:52:24 2023

@author: Xinyang Chen
"""
from utils import plot_one_contourf
import numpy as np

ploter = lambda X, dx, dy: (plot_one_contourf(X.transpose(), dx, dy, "temperature", "temperature[◦C]", 0.0, 1.0))
MAE = lambda array1, array2, domain: np.linalg.norm(array1[domain] - array2[domain])

lambda_list = [MAE, ploter]