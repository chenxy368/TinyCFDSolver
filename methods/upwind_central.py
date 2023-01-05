# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 07:38:52 2022

@author: Xinyang Chen
"""
from solvers.upwind_central_solver import UpwindCentral2DSolver
from utils.grid_loader import GridLoader

class UpwindCentral2D():
    """Upwind Central Scheme Method
    
    Implementation of upwind central scheme Method for 2D advection diffusion equation.
    See discription and formulas at https://en.wikipedia.org/wiki/Upwind_scheme
    
    Attributes:
        shape: shape of the grid
        X_boundaries: boundary objects
        method_name: name of the mehod, should be UpwindCentral
        solver: an upwind central scheme solver
    """
    def __init__(self, root, step_visualization = None, final_visualization = None, 
                 initial_condition = None):
        """ Inits UpwindCentral2D class with root of the project, step and final visulization lambda functions and 
        initial conditions"""
        loader = GridLoader(root)
        
        domain_dict = {
            "mesh": "mesh.csv",
        }
        mesh_boundary_dict ={
            "mesh": ["X"],
        }
        
        method_info, mesh_data, _ = loader.load_grid(domain_dict, mesh_boundary_dict)
        self.shape = mesh_data[0]["mesh"]
        self.x_boundaries = mesh_data[3]["X"]
        
        self.method_name = method_info[0]


        self.solver = UpwindCentral2DSolver(self.shape, 
                                            (mesh_data[1]["dx"], mesh_data[1]["dy"], mesh_data[1]["dt"], \
                                             mesh_data[1]["u"], mesh_data[1]["v"], mesh_data[1]["molecular_diffusivity_x"], \
                                             mesh_data[1]["molecular_diffusivity_y"]), 
                                            (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                            self.boundary_process, self.extra_computing,
                                            step_visualization, final_visualization, initial_condition)  
    
    def solve(self, num_timesteps, checkpoint_interval): 
        """ Call solver's solve function
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result from solver
        """
        return self.solver.solve(num_timesteps, checkpoint_interval)
    
    """
    Boundary processing functions, get variable from solver and process with the boundaries and send back
    """
    def boundary_process(self, X, t):
        for boundary in self.x_boundaries:
            X = boundary.process(X)
        return X
    
    def extra_computing(self, X, t):
        pass