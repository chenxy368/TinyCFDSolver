# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 23:55:41 2022

@author: Xinyang Chen
"""
from utils.grid_loader import GridLoader
from solvers.crank_nicolson_2D_solver import CrankNicolson2DSolver
import numpy as np

time_boundaries = ["Implicit Linear 2D Time"]
implcit_boundry_classes = ["Implicit Linear 2D", "Implicit Linear 2D Time"]

class CrankNicolson2D():
    """Crank-Nicolson 2D Method
    
    Implementation of Crank-Nicolson method for 2D diffusion equation.
    See discription and formulas at https://en.wikipedia.org/wiki/Diffusion_equation
    
    Attributes:
        shape: shape of the grid
        x_boundaries: the boundary objects
        x_implicit_boundaries: the implicit boundary objects(provide process_RHS and modify_matrix methods)
        method_name: name of the mehod, should be CrankNicolson2D
        solver: a Crank-Nicolson solver
    """
    def __init__(self, root, lambda_list: list, initial_condition = None):
        """ Inits CrankNicolson2D class with root of the project, lambda functions list,
        and initial conditions"""

        step_visualization = None
        final_visualization = None
        if len(lambda_list) > 0:
            step_visualization = lambda_list[0]
        if len(lambda_list) > 1:
            final_visualization = lambda_list[1]

        # Load Grid
        loader = GridLoader(root)
        
        domain_dict = {
            "mesh": "mesh.csv",
        }
        mesh_boundary_dict ={
            "mesh": ["X"],
        }
        
        method_info, mesh_data, mesh_dict = loader.load_grid(domain_dict, mesh_boundary_dict)

        
        params = (mesh_data[1]["dx"], mesh_data[1]["dy"], mesh_data[1]["dt"], mesh_data[1]["alpha"])
        Q, self.x_boundaries, self.x_implicit_boundaries, self.x_time_boundaries = \
            self.init_matrices(mesh_dict["mesh"], params, mesh_data[3]["X"], mesh_data[2]["mesh"])

        self.solver = CrankNicolson2DSolver(mesh_data[0]["mesh"], Q, params, (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                            self.boundary_process, self.extra_computing, step_visualization, final_visualization, initial_condition)
    
    def init_matrices(self, mesh: np.ndarray, params: tuple, boundaries: list, 
                      interior: tuple):
        """ Compute Q matrix in linear system Q = bx and reset boundary condition
        
            Since some nodes are not included in the implicit solving process, they are deleted from the
            original Q. Correspondingly, they right hand side also change which influences the RHS_domain
            in the boundary classes. Therefore, the RHS_domains are reseted based on new right hand side.
        Args:
            mesh: the mesh array
            params: dx, dy, dt and alpha
            shape: shape of the grid
            boundaries: boundary objects
        Return:
            result from solver
        """
        shape = mesh.shape
        
        # Filter out explicit boundary classes which don't modify the matrix and right hand side value
        implicit = []
        explicit = []
        time = []
        for boundary in boundaries:
            if boundary.get_type() in implcit_boundry_classes:
                implicit.append(boundary)
            else:
                explicit.append(boundary)

            # Regist time boundaries
            if boundary.get_type() in time_boundaries:                
                boundary.set_dt(params[2])
                time.append(boundary)
        
        # Computing params
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
        
        # Reset boundary classes' right hand side domain
        interior_list = []
        for i in range(len(interior[0])):
            interior_list.append(interior[1][i] * Nx + interior[0][i])

        interior_list.sort()
        for boundary in implicit:
            Q = boundary.modify_matrix(Q)
            boundary.modify_RHS_domain(np.array(interior_list))
        
        Q = Q[:, interior_list]
        Q = Q[interior_list]
        return Q, explicit, implicit, time
        
    def solve(self, params: list): 
        """ Call solver's solve function
        Args:
            params[0]: num_timesteps, the number of total timesteps
            params[1]: checkpoint_interval, frequency of calling step postprocess
        Return:
            result from solver
        """
        return self.solver.solve(int(params[0]), params[1])
        
    """
    Boundary processing functions, get variable from solver and process with the boundaries and send back
    """
    def boundary_process(self, X, RHS, t):
        for boundary in self.x_boundaries:
            X = boundary.process(X)
        for boundary in self.x_implicit_boundaries:   
            X = boundary.process(X)
            RHS = boundary.process_RHS(RHS)
        # Reset boundary params based on time for time variant boundaries
        for boundary in self.x_time_boundaries:
            boundary.reset_curr_params(t)
        
        return X, RHS
        
    def extra_computing(self, X, t):
        pass