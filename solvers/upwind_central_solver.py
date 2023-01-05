# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 05:24:50 2022

@author: Xinyang Chen
"""
import numpy as np

class UpwindCentral2DSolver():
    """Upwind Central Scheme Method
    
    Implementation of upwind central scheme Method for 2D advection diffusion equation. Solver is called in method.
    See discription and formulas at https://en.wikipedia.org/wiki/Upwind_scheme
    
    Attributes:
        shape: shape of the grid
        dt, dx, dy: timestep length, grid length
        coefficients: coefficient of different terms
        interior: interior slice of grid
        exterior: exterior slice of grid
        boundary_process: boundaries process functions
        extra_computing: extra computing at the end of each timestep
        step_visualization: visualization function at each timestep
        final_visualization: final visualization function
        initial_condition: initial conditions
    """
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process,
                 extra_computing = None, step_visualization = None, final_visualization = None, 
                 initial_condition = None):
        self.shape = shape
        self.dt, self.dx, self.dy, self.coefficients = self.compute_upwind_param(params)
        
        self.interior = domains[0]
        self.exterior = domains[1]

        self.boundary_process = boundary_process
        self.extra_computing = extra_computing

        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
        
        self.initial_condition = initial_condition
    
    def compute_upwind_param(self, params: list):
        """ Compute coeffiecients of terms based on upwind scheme
       
            X[i, j] = self.coefficients[0] * X[i, j] + 
                      self.coefficients[1] * X[i-1, j] +
                      self.coefficients[2] * X[i, j-1] +
                      self.coefficients[3] * X[i+1, j] +
                      self.coefficients[4] * X[i, j+1]
            keep the advection term upwind
            see referenc at https://computationalthinking.mit.edu/Spring21/2d_advection_diffusion/
            note that all coeffcient are const in this method (linear method)
        
        Args:
            params: list [dx, dy, dt, u, v, mu_x, mu_y]
        Return:
            result tuple (u, v, p)
        """
        dx = params[0]
        dy = params[1]
        dt = params[2]
        u = params[3]
        v = params[4]
        mu_x = params[5]
        mu_y = params[6]
        ax = u * (dt / dx)
        ay = v * (dt / dy)
        betax = mu_x * (dt / dx / dx)
        betay = mu_y * (dt / dy / dy)
        
        if u > 0:
            coefficient2 = ax + betax
            coefficient4 = betax
        else:
            coefficient2 = betax
            coefficient4 = betax - ax
            
        if v > 0:
            coefficient3 = ay + betay
            coefficient5 = betay
        else:
            coefficient3 = betay
            coefficient5 = betay - ay
        
        return dt, dx, dy, (1 - (u / abs(u)) * ax - (v / abs(v)) * ay - 2 * betax - 2 * betay, \
                coefficient2, coefficient3, coefficient4, coefficient5)
            
    
    def solve(self, num_timesteps, checkpoint_interval):
        """ Solve advection diffusion equation with upwind central scheme method
        
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result X
        """
        
        # Initialize 
        X_list = []
        X = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        if self.initial_condition is not None:
            X[...] = self.initial_condition[...]

        for t in range(num_timesteps):
            # Impose boundary conditions
            X = self.boundary_process(X, t)
            
            # Compute u @ n+1
            X[self.interior] = self.coefficients[0] * X[self.interior] + \
                               self.coefficients[1] * X[self.interior[0] - 1, self.interior[1]] + \
                               self.coefficients[2] * X[self.interior[0], self.interior[1] - 1] + \
                               self.coefficients[3] * X[self.interior[0] + 1, self.interior[1]] + \
                               self.coefficients[4] * X[self.interior[0], self.interior[1] + 1]
            
            self.extra_computing(X, t)
            if self.final_visualization is not None and (t + 1) % checkpoint_interval == 0:
                X_list.append((np.transpose(X)))
                
                if self.step_visualization is not None:
                    self.step_visualization(X, self.dx, self.dy, self.dt, t)

        if self.final_visualization is not None:
            self.final_visualization(X_list, self.dx, self.dy)
            
        return X
    