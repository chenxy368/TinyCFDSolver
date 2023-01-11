# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 01:18:58 2022

@author: Xinyang Chen
"""
import numpy as np

class CrankNicolson2DSolver():
    """Crank-Nicolson 2D Solver
    
    Implementation of Crank-Nicolson method for diffusion equation. Solver is called in method.
    See discription and formulas at https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method
    
    Attributes:
        shape: the shape of grid
        dt, dx, dy, alpha: timestep length, grid length, diffusion coefficient 
        interior: interior slice of grid
        exterior: exterior slice of grid
        boundary_process: boundaries process functions
        extra_computing: extra computing at the end of each timestep
        step_visualization: visualization function at each timestep
        final_visualization: final visualization function
        initial_condition: initial conditions
    """
    def __init__(self, shape: tuple, Q: np.ndarray, params: list, domains: list, boundary_process,
                 extra_computing = None, step_visualization = None, final_visualization = None, 
                 initial_condition = None):
        self.shape = shape
        
        # Linear System: Qx = b
        self.Q = Q
        
        self.dx = params[0]
        self.dy = params[1]
        self.dt = params[2]
        self.alpha = params[3]

        self.interior = domains[0]
        self.exterior = domains[1]

        self.boundary_process = boundary_process
        self.extra_computing = extra_computing

        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
        
        self.initial_condition = initial_condition
        
    
    def solve(self, num_timesteps, checkpoint_interval):
        """ Solve diffusion equation with Crank-Nicolson method
        
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result
        """
        betax = self.alpha * self.dt / (2 * self.dx * self.dx)
        betay = self.alpha * self.dt / (2 * self.dy * self.dy)
        
        X_list = []
        # Initialize
        X = np.zeros([self.shape[0], self.shape[1]], dtype = float)
       
        # Initial conditions
        if self.initial_condition is not None:
            X[...] = self.initial_condition[...]   
        
        array_to_tuple = [(self.interior[0][i], self.interior[1][i]) for i in range(len(self.interior[0]))]
        sorted_list = sorted(array_to_tuple, key=lambda t: (t[1], t[0]))
        RHS_interior0 = np.zeros([len(self.interior[0])], dtype = int)
        RHS_interior1 = np.zeros([len(self.interior[1])], dtype = int)
        for i in range(len(self.interior[0])):
            RHS_interior0[i] = sorted_list[i][0]
            RHS_interior1[i] = sorted_list[i][1]
        RHS_interior = (RHS_interior0, RHS_interior1)
        
        # Time loop
        for t in range(num_timesteps):
            print("Timestep: " + str(t))
            # Impose boundary conditions
            b = np.zeros([self.Q.shape[0]], dtype = float)
            X, b = self.boundary_process(X, b, t)

            b[:] += (1 - 2 * betax - 2 * betay) * X[RHS_interior] + \
                   betax * X[RHS_interior[0] - 1, RHS_interior[1]] + betax * X[RHS_interior[0] + 1, RHS_interior[1]] + \
                   betay * X[RHS_interior[0], RHS_interior[1] - 1] + betay * X[RHS_interior[0], RHS_interior[1] + 1]


            b = np.reshape(b, b.shape + (1,))
            
            # Solve Qx = b
            b = np.linalg.solve(self.Q, b)
                    
            X[RHS_interior] = b[:, 0]
            
            self.extra_computing(X, t)
            if self.final_visualization is not None and (t + 1) % checkpoint_interval == 0:
                X_list.append(np.transpose(X).copy())
                
                if self.step_visualization is not None:
                    self.step_visualization(X, self.dx, self.dy, self.dt, t)

        if self.final_visualization is not None:
            self.final_visualization(X_list, self.dx, self.dy)

        return X