# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 23:55:41 2022

@author: HP
"""
from utils.grid_loader import GridLoader
from solvers.crank_nicolson_2D_solver import CrankNicolson2DSolver
import numpy as np

class CrankNicolson2D():
    def __init__(self, root, step_visualization = None, final_visualization = None, 
                 initial_condition = None):
        loader = GridLoader(root)
        
        domain_dict = {
            "mesh": "mesh.csv",
        }
        mesh_boundary_dict ={
            "mesh": ["X"],
        }
        
        method_info, mesh_data, mesh_dict = loader.load_grid(domain_dict, mesh_boundary_dict)
        
        self.shape = mesh_data[0]["mesh"]
        
        params = (mesh_data[1]["dx"], mesh_data[1]["dy"], mesh_data[1]["dt"], mesh_data[1]["alpha"])
        Q, self.x_boundaries = self.init_matrices(mesh_dict["mesh"], params, self.shape, mesh_data[3]["X"], mesh_data[2]["mesh"])
        
        self.solver = CrankNicolson2DSolver(self.shape, Q, params, (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                            self.boundary_process, self.extra_computing, step_visualization, final_visualization, initial_condition)
    
    def init_matrices(self, mesh: np.ndarray, params: tuple, shape: tuple, boundaries: list, 
                      interior: tuple):
        dx = params[0]
        dy = params[1]
        dt = params[2]
        alpha = params[3]
        betax = alpha * dt / (2 * dx * dx)
        betay = alpha * dt / (2 * dy * dy)
        
        # Define matrix Block matrices A, B, C
        Nx = shape[0]
        Ny = shape[1]
        A = np.zeros([Nx, Nx], dtype = float)
        B = np.zeros([Nx, Nx], dtype = float)
        C = np.zeros([Nx, Nx], dtype = float)

        B[0, 0] = 1 + 2 * betax + 2 * betay
        B[0, 1] = -betax
        B[-1, -2] = -betax
        B[-1, -1] = 1 + 2 * betax + 2 * betay
        for i in range(1, Nx-1):
            B[i, i-1] = -betax
            B[i, i] = 1 + 2 * betax + 2 * betay
            B[i, i+1] = -betax

        for i in range(Nx):
            A[i, i] = -betay;
            C[i, i] = -betay;

        # Define matrix Q
        Q = np.zeros([Nx * Ny, Nx * Ny], dtype = float)
        Q[0 : Nx, 0 : Nx] = B[:, :]
        Q[0 : Nx, Nx : 2*Nx] = C[:, :]
        Q[(Ny-1)*Nx :, (Ny-2)*Nx : (Ny-1)*Nx] = A[:, :]
        Q[(Ny-1)*Nx :, (Ny-1)*Nx :] = B[:, :]
        for i in range(1, Ny-1):
            Q[i*Nx : (i+1)*Nx, (i-1)*Nx : i*Nx] = A[:, :]
            Q[i*Nx : (i+1)*Nx, i*Nx : (i+1)*Nx] = B[:, :]
            Q[i*Nx : (i+1)*Nx, (i+1)*Nx : (i+2)*Nx] = C[:, :]
        
        interior_list = []
        for i in range(len(interior[0])):
            interior_list.append(interior[1][i] * Nx + interior[0][i])

        interior_list.sort()
        for boundary in boundaries:
            Q = boundary.modify_matrix(Q)
            boundary.modify_RHS_domain(np.array(interior_list))
        
        Q = Q[:, interior_list]
        Q = Q[interior_list]
        return Q, boundaries
        
    def solve(self, num_timesteps, checkpoint_interval): 
        return self.solver.solve(num_timesteps, checkpoint_interval)
        
    def boundary_process(self, X, RHS, t):
        for boundary in self.x_boundaries:
            X = boundary.process(X)
            RHS = boundary.process_RHS(RHS)
        
        return X, RHS
        
    def extra_computing(self, X, t):
        pass