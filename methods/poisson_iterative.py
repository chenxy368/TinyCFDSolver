# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 03:58:31 2022

@author: Xinyang Chen
"""
from solvers.poisson_iterative_solver import PointJacobiSolver, GaussSeidelSolver, SORSolver
from utils.grid_loader import GridLoader
import numpy as np
import os

class PoissonIterative():
    """Poission iterative solving Method
    
    Implementation of Possion iterative method for Poisson equation.
    See discription and formulas at https://en.wikipedia.org/wiki/Poisson%27s_equation
    
    Attributes:
        shape: the shape of the matrix
        x_boundaries: the boundary objects
        p_boundaries: the boundary objects of p
        method_name: name of the mehod, should be PoissonIterative
        solver: a poisson iterative solver
    """
    def __init__(self, root, lambda_list: list, initial_condition = None):
        """ Inits PoissonIterative class with root of the project, lambda functions list and initial condition"""
        
        # Assert erorr metrics
        assert len(lambda_list) > 0 and callable(lambda_list[0]) and lambda_list[0].__name__ == "<lambda>" 
        
        metrics = lambda_list[0]
        final_visualization = None
        if len(lambda_list) > 1:
            final_visualization = lambda_list[1]
        
        # Load grid
        loader = GridLoader(root)
        
        domain_dict = {
            "mesh": "mesh.csv",
        }
        mesh_boundary_dict ={
            "mesh": ["X"],
        }

        method_info, mesh_data, _ = loader.load_grid(domain_dict, mesh_boundary_dict)
        self.shape = mesh_data[0]["mesh"]
        
        # Get boundary
        self.x_boundaries = mesh_data[3]["X"]
        
        # Get method name
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

    def solve(self, params: list):
        """ Call solver's solve function
        Args:
            params: the right hand side of Poisson equation, an float or a path to numpy array
        Return:
            result from solver, a two dimensional numpy array
        """
        if type(params[0]) is not np.ndarray:
            f = np.full((self.shape[0], self.shape[1]), float(params[0]), dtype = float)
        else:
            if not os.path.exists(params[0]):
                raise RuntimeError(params[0] + "do not exist")
            f = np.load(params[0])
            
        return self.solver.solve(f)
    
    """
    Boundary processing function, get variable from solver and process with the boundaries and send back
    """
    def boundary_process(self, x):
        for boundary in self.x_boundaries:
            x = boundary.process(x)

        return x
