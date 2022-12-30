# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 03:58:31 2022

@author: HP
"""
from solvers.poisson_iterative_solver import PointJacobiSolver, GaussSeidelSolver, SORSolver
from utils.grid_loader import GridLoader
import numpy as np

class PoissonIterative():
    def __init__(self, root, metrics = None, final_visualization = None, initial_condition = None):
        assert callable(metrics) and metrics.__name__ == "<lambda>" 
        
        loader = GridLoader(root)
        
        domain_dict = {
            "mesh": "mesh.csv",
        }
        mesh_boundary_dict ={
            "mesh": ["X"],
        }

        method_info, mesh_data = loader.load_grid(domain_dict, mesh_boundary_dict)
        self.shape = mesh_data[0]["mesh"]
        self.x_boundaries = mesh_data[3]["X"]
        
        self.method_name = method_info[0]
        # Poisson iterative solver
        solver_name = method_info[1]

        if solver_name == "SOR":
            self.solver = SORSolver(self.shape, 
                                       (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                       (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                       self.boundary_process, float(method_info[2]), metrics, 
                                       float(method_info[3]), final_visualization, initial_condition)  
        elif solver_name == "GaussSeidel":
            self.solver = GaussSeidelSolver(self.shape, 
                                               (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                               (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                               self.boundary_process, float(method_info[2]), metrics, 
                                               final_visualization, initial_condition)
        else:
            self.solver = PointJacobiSolver(self.shape, 
                                               (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                               (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                               self.boundary_process, float(method_info[2]), metrics, 
                                               final_visualization, initial_condition)

    def solve(self, f):
        if type(f) is not np.ndarray:
            f = np.full((self.shape[0], self.shape[1]), float(f), dtype = float)
        return self.solver.solve(f)
    
    def boundary_process(self, x):
        for boundary in self.x_boundaries:
            x = boundary.process(x)

        return x
