# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 05:03:39 2023

@author: Xinyang Chen
"""
from utils.grid_loader import GridLoader
from utils.implicit_boundary import ImplicitLinearCondition1D, ImplicitLinearCondition1DTime
from solvers.ADI_solver import ADISolver
import numpy as np

# The boundary classes need timestep to reset parameters
time_boundaries = ["Implicit Linear 2D Time", "Implicit Linear 1D Time"]

# The boundary classes need reduce to 1D for implicit computing based on ADI
boundaries_2D_to_1D = {"Implicit Linear 2D": ImplicitLinearCondition1D,
                       "Implicit Linear 2D Time": ImplicitLinearCondition1DTime}

class ADI():
    """Alternating-direction implicit Method
    
    Implementation of alternating-direction implicit method for diffusion equation.
    See discription and formulas at https://en.wikipedia.org/wiki/Alternating-direction_implicit_method
    
    Attributes:
        x_boundaries: the boundary objects
        x_time_boundaries: the boundary objects need timestep to reset parameters
        method_name: name of the mehod, should be FracStep
        TDMA: use TDMA to solve linear system or not
        solver: an ADI solver
    """
    def __init__(self, root, lambda_list: list, initial_condition = None):
        """ Inits ADI class with root of the project, lambda functions list, and initial conditions"""
        
        step_visualization = None
        final_visualization = None
        if len(lambda_list) > 0:
            step_visualization = lambda_list[0]
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
        
        method_info, mesh_data, mesh_dict = loader.load_grid(domain_dict, mesh_boundary_dict)
        
        # Get method name
        self.method_name = method_info[0]
        
        # Check TDMA flag
        if len(method_info) > 1 and method_info[1] == "TDMA":
            self.TDMA = True
        else:
            self.TDMA = False
        
        # Get boundary
        self.x_boundaries = mesh_data[3]["X"]
        
        # Get params, problem sets, and time boundaries
        params = (mesh_data[1]["dx"], mesh_data[1]["dy"], mesh_data[1]["dt"], mesh_data[1]["alpha"])
        problem_set_x, problem_set_y, self.x_time_boundaries = self.init_matrices(mesh_dict["mesh"], params, mesh_data[3]["X"], mesh_data[2]["mesh"])
        
        # Init solver
        self.solver = ADISolver(mesh_data[0]["mesh"], problem_set_x, problem_set_y, params,  
                                (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                self.boundary_process_2D, self.boundary_process_RHS, self.TDMA, self.extra_computing, 
                                step_visualization, final_visualization, initial_condition)
    
    def init_matrices(self, mesh: np.ndarray, params: tuple, boundaries: list, interior: tuple):
        """ Generate problem sets for ADI solver
        
        In brief, the ADI method solve two dimmensional problems by first solving along x direction and then solving
        along y direction. This method generate all one dimmensional problems along two directions.
        
        Args:
            mesh: the grid array
            params: parametere including dx, dy, dt, and alpha
            boundaries: boundary objects
            interior: interior slice of the mesh
        Return:
            problem_set_x: problem sets along x direction
            problem_set_y: problem sets along y direction
            time: boundary objects need timestep to reset parameters
        """
        # Get shape
        shape = mesh.shape
        
        # Compute parameters
        dx = params[0]
        dy = params[1]
        dt = params[2]
        alpha = params[3]
        betax = alpha * dt / (dx * dx)
        betay = alpha * dt / (dy * dy)

        # Process boundary objects
        boundary_dict = {}     
        time = []
        for boundary in boundaries:
            boundary_dict[boundary.get_id()] = boundary
            if boundary.get_type() in time_boundaries:
                boundary.set_dt(dt)
                time.append(boundary)
        
        # Get interior_ID
        interior_ID = mesh[interior[0][0], interior[1][0]]
        
        # Get problem sets along x direction
        problem_set_x = []
        for y in range(shape[1]):
            tmp = mesh[:, y]
            # Skip if no interior nodes on this row
            if np.where(tmp == interior_ID)[0].shape[0] == 0:
                continue
            
            # Get corresponding 1D boundary objects
            curr_boundries = []
            for x in range(shape[0]):
                if tmp[x] in boundary_dict and boundary_dict[tmp[x]].get_type() in boundaries_2D_to_1D:
                    boundary = boundary_dict[tmp[x]]
                    params = boundary.get_params_x()
                    # Skip if do not need process
                    if params[2] == 0:
                        continue
                    curr_boundries.append(boundaries_2D_to_1D[boundary.get_type()](tmp[x], boundary.get_name(), x, params))
                    if curr_boundries[len(curr_boundries) - 1].get_type() in time_boundaries:
                        curr_boundries[len(curr_boundries) - 1].set_dt(dt)
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
            
            # Get all interior indexes
            interior_arr = np.where(tmp == interior_ID)

            # Reset right hand side domain
            for boundary in curr_boundries:
                Ax = boundary.modify_matrix(Ax)
                boundary.modify_RHS_domain(interior_arr)
                
            # Cut Ax
            Ax = Ax[:, interior_arr[0]]
            Ax = Ax[interior_arr[0]]

            # Assembly problem set
            if self.TDMA:
                # For TDMA generate subdiagonal, diagonal, and supdiagonal vectors
                subdiagonal = np.zeros([Ax.shape[0]], dtype = float)
                superdiagonal = np.zeros([Ax.shape[0]], dtype = float)
                diagonal = np.zeros([Ax.shape[0]], dtype = float)
                subdiagonal[0] = 0
                subdiagonal[Ax.shape[0] - 1] = Ax[Ax.shape[0] - 1, Ax.shape[0] - 2]
                diagonal[0] = Ax[0, 0]
                diagonal[Ax.shape[0] - 1] = Ax[Ax.shape[0] - 1, Ax.shape[0] - 1]
                superdiagonal[0] = Ax[0, 1]
                superdiagonal[Ax.shape[0] - 1] = 0
                for i in range(1, Ax.shape[0] - 1):
                    subdiagonal[i] = Ax[i, i-1]
                    diagonal[i] = Ax[i, i]
                    superdiagonal[i] = Ax[i, i+1]
                problem_set_x.append([curr_boundries, (subdiagonal, diagonal, superdiagonal), interior_arr, y])
            else:
                problem_set_x.append([curr_boundries, Ax, interior_arr, y])
            
        # Get problem set on y direction
        problem_set_y = []
        for x in range(shape[0]):
            tmp = mesh[x, :]
            if np.where(tmp == interior_ID)[0].shape[0] == 0:
                continue
            
            curr_boundries = []
            for y in range(shape[1]):
                if tmp[y] in boundary_dict and boundary_dict[tmp[y]].get_type() in boundaries_2D_to_1D:
                    boundary = boundary_dict[tmp[y]]
                    params = boundary.get_params_y()
                    if params[2] == 0:
                        continue
                    curr_boundries.append(boundaries_2D_to_1D[boundary.get_type()](tmp[y], boundary.get_name(), y, params))
                    if curr_boundries[len(curr_boundries) - 1].get_type() in time_boundaries:
                        curr_boundries[len(curr_boundries) - 1].set_dt(dt)
                        
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

            if self.TDMA:
                subdiagonal = np.zeros([Ay.shape[0]], dtype = float)
                superdiagonal = np.zeros([Ay.shape[0]], dtype = float)
                diagonal = np.zeros([Ay.shape[0]], dtype = float)
                subdiagonal[0] = 0
                subdiagonal[Ay.shape[0] - 1] = Ay[Ay.shape[0] - 1, Ay.shape[0] - 2]
                diagonal[0] = Ay[0, 0]
                diagonal[Ay.shape[0] - 1] = Ay[Ay.shape[0] - 1, Ay.shape[0] - 1]
                superdiagonal[0] = Ay[0, 1]
                superdiagonal[Ay.shape[0] - 1] = 0
                
                for i in range(1, Ay.shape[0] - 1):
                    subdiagonal[i] = Ay[i, i-1]
                    diagonal[i] = Ay[i, i]
                    superdiagonal[i] = Ay[i, i+1]
                problem_set_y.append([curr_boundries, (subdiagonal, diagonal, superdiagonal), interior_arr, x])
            else:
                problem_set_y.append([curr_boundries, Ay, interior_arr, x])        

        return problem_set_x, problem_set_y, time
                    
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
    def boundary_process_2D(self, X, t):
        for boundary in self.x_time_boundaries:
            boundary.reset_curr_params(t)
            X = boundary.process(X)
        for boundary in self.x_boundaries:
            X = boundary.process(X)

        return X
    
    def boundary_process_RHS(self, b, t, boundaries):
        for boundary in boundaries:
            if boundary.get_type() in time_boundaries:
                boundary.reset_curr_params(t)
            b = boundary.process_RHS(b)

        return b
    
    def extra_computing(self, X, t):
        pass
        
        