# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 05:37:11 2022

@author: HP
"""
from solvers.frac_step_solver import FracStepSolver
from solvers.poisson_iterative_solver import PointJacobiSolver, GaussSeidelSolver, SORSolver
from solvers.streamfunction_vorticity_solver import StreamFunctionVorticitySolver
from solvers.upwind_central_solver import UpwindCentral2DSolver
from solvers.crank_nicolson_2D_solver import CrankNicolson2DSolver
from solvers.ADI_solver import ADISolver