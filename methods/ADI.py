# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 05:03:39 2023

@author: HP
"""
from utils.grid_loader import GridLoader
from utils.implicit_boundary import FirstOrderCondition1D
from solvers.ADI_solver import ADISolver
import numpy as np

boundaries_2D_to_1D = {"Implicit First Order 2D": FirstOrderCondition1D}

class ADI():
    def __init__(self, root, step_visualization = None, final_visualization = None, initial_condition = None):
        
        loader = GridLoader(root)
        
        domain_dict = {
            "mesh": "mesh.csv",
        }
        mesh_boundary_dict ={
            "mesh": ["X"],
        }
        
        method_info, mesh_data, mesh_dict = loader.load_grid(domain_dict, mesh_boundary_dict)
        
        self.shape = mesh_data[0]["mesh"]
        self.x_boundaries = mesh_data[3]["X"]
        
        params = (mesh_data[1]["dx"], mesh_data[1]["dy"], mesh_data[1]["dt"], mesh_data[1]["alpha"])
        problem_set_x, problem_set_y = self.init_matrices(mesh_dict["mesh"], params, self.shape, mesh_data[3]["X"], mesh_data[2]["mesh"])
        self.problem_set_x = problem_set_x
        self.problem_set_y = problem_set_y
        
        self.solver = ADISolver(self.shape, problem_set_x, problem_set_y, params,  
                                (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                self.boundary_process_2D, self.boundary_process_RHS, self.extra_computing, 
                                step_visualization, final_visualization, initial_condition)
    
    def init_matrices(self, mesh: np.ndarray, params: tuple, shape: tuple, boundaries: list, 
                      interior: tuple):
        dx = params[0]
        dy = params[1]
        dt = params[2]
        alpha = params[3]
        betax = alpha * dt / (dx * dx)
        betay = alpha * dt / (dy * dy)

        boundary_dict = {}        
        for boundary in boundaries:
            boundary_dict[boundary.get_id()] = boundary
        
        interior_ID = mesh[interior[0][0], interior[1][0]]
        
        problem_set_x = []
        for y in range(shape[1]):
            tmp = mesh[:, y]
            if np.where(tmp == interior_ID)[0].shape[0] == 0:
                continue
            
            curr_boundries = []
            for x in range(shape[0]):
                if tmp[x] in boundary_dict and boundary_dict[tmp[x]].get_type() in boundaries_2D_to_1D:
                    boundary = boundary_dict[tmp[x]]
                    params = boundary.get_params()
                    if params[1][1] != 0:
                        continue
                    params[1] = params[1][0]
                    curr_boundries.append(boundaries_2D_to_1D[boundary.get_type()](tmp[x], boundary.get_name(), x, params))
                    
            # Define matrix Ax
            Ax = np.zeros([shape[0], shape[0]], dtype = float)

            Ax[0, 0] = 1 + 2 * betax
            Ax[0, 1] = -betax
            Ax[-1, -2] = -betax
            Ax[-1, -1] = 1 + 2 * betax
            for i in range(1, shape[0] - 1):
                Ax[i, i-1] = -betax
                Ax[i, i] = 1 + 2 * betax
                Ax[i, i+1] = -betax
            
            interior_arr = np.where(tmp == interior_ID)

            for boundary in curr_boundries:
                Ax = boundary.modify_matrix(Ax)
                boundary.modify_RHS_domain(interior_arr)
                
            Ax = Ax[:, interior_arr[0]]
            Ax = Ax[interior_arr[0]]

            problem_set_x.append([curr_boundries, Ax, interior_arr, y])
            
 
        problem_set_y = []
        for x in range(shape[0]):
            tmp = mesh[x, :]
            if np.where(tmp == interior_ID)[0].shape[0] == 0:
                continue
            
            curr_boundries = []
            for y in range(shape[1]):
                if tmp[y] in boundary_dict and boundary_dict[tmp[y]].get_type() in boundaries_2D_to_1D:
                    boundary = boundary_dict[tmp[y]]
                    params = boundary.get_params()
                    if params[1][0] != 0:
                        continue
                    params[1] = params[1][1]
                    curr_boundries.append(boundaries_2D_to_1D[boundary.get_type()](tmp[y], boundary.get_name(), y, params))
                    
            # Define matrix Ay
            Ay = np.zeros([shape[1], shape[1]], dtype = float)

            Ay[0, 0] = 1 + 2 * betay
            Ay[0, 1] = -betay
            Ay[-1, -2] = -betay
            Ay[-1, -1] = 1 + 2 * betay
            for i in range(1, shape[1] - 1):
                Ay[i, i-1] = -betay
                Ay[i, i] = 1 + 2 * betay
                Ay[i, i+1] = -betay
            
            interior_arr = np.where(tmp == interior_ID)

            for boundary in curr_boundries:
                Ay = boundary.modify_matrix(Ay)
                boundary.modify_RHS_domain(interior_arr)
 
            Ay = Ay[:, interior_arr[0]]
            Ay = Ay[interior_arr[0]]
            
            problem_set_y.append((curr_boundries, Ay, interior_arr, x))        

        return problem_set_x, problem_set_y
                    
    def solve(self, num_timesteps, checkpoint_interval): 
        return self.solver.solve(num_timesteps, checkpoint_interval)
        
    def boundary_process_2D(self, X, t):
        for boundary in self.x_boundaries:
            X = boundary.process(X)

        return X
    
    def boundary_process_RHS(self, b, t, boundaries):
        for boundary in boundaries:
            b = boundary.process_RHS(b)
        
        return b
    
    def extra_computing(self, X, t):
        pass
        
        