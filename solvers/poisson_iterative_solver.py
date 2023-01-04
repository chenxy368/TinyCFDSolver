# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:48:39 2022

@author: Xinyang Chen
"""
import numpy as np

class PoissonIterativeSolver():
    """Poisson Iterative Solver Base Class
        
    The base class of different Poisson iterative solvers
        
    Attributes:
        shape: the shape of  grid
        dx, dy, mu, rho: grid length
        interior: interior slice of grid
        exterior: exterior slice of grid
        tol: convergence tolerance
        extra_computing: extra computing at the end of each timestep
        step_visualization: visualization function at each timestep
        initial_condition: initial condition
    """
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process, tol, metrics = None, 
                 final_visualization = None, initial_condition = None):
        assert callable(metrics) and metrics.__name__ == "<lambda>"

        # Check Initial Condition's Shape
        if initial_condition is not None and initial_condition.shape != shape:
            raise RuntimeError("Initial Condition Shape Dismatch")
        
        self.shape = shape
        self.dx = params[0]
        self.dy = params[1]
        
        self.interior = domains[0]
        self.exterior = domains[1]

        self.boundary_process = boundary_process

        self.tol = tol
        
        self.metrics = metrics
        
        self.final_visualization = final_visualization
        
        self.initial_condition = initial_condition
        
class PointJacobiSolver(PoissonIterativeSolver):
    """Point Jacobi Solver Class
        
    Implementation of Point Jacobi Method. See reference at https://en.wikipedia.org/wiki/Jacobi_method
    """
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process, tol, metrics = None,
                 final_visualization = None, initial_condition = None):
        super(PointJacobiSolver, self).__init__(shape, params, domains, boundary_process, tol, metrics, final_visualization)
        
    def solve(self, f):
        """ Solve Poissson equation with Point Jacobi Method
        
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result after converging
        """
        # Initialization
        X = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        if self.initial_condition is not None:
            X[...] = self.initial_condition[...]
            
        iteration = 0
        error = 1 

        # Solve iteratively
        while error > self.tol:
            X = self.boundary_process(X)
                    
            tmp = np.zeros_like(X)
            tmp[self.interior] = (1.0 / (2.0 / self.dx / self.dx + 2.0 / self.dy / self.dy)) * ((1.0 / self.dx / self.dx) * X[self.interior[0] - 1, self.interior[1]] \
                                + (1.0 / self.dx / self.dx) * X[self.interior[0] + 1, self.interior[1]] + (1.0 / self.dy / self.dy) * X[self.interior[0], self.interior[1] - 1] \
                                    + (1.0 / self.dy / self.dy) * X[self.interior[0], self.interior[1] + 1] - f[self.interior])
                
            error = self.metrics(tmp, X, self.interior)
                
            X[self.interior] = tmp[self.interior]
                
            iteration += 1
        
        # Postprocess
        print('Number of iterations in Possion Iterative Solver: ', iteration)
        
        if self.final_visualization is not None:
            self.final_visualization(X, self.dx, self.dy)
        
        return X
        
class GaussSeidelSolver(PoissonIterativeSolver):
    """Gauss Seidel Solver Class
        
    Implementation of Gauss Seidel Method. See reference at https://en.wikipedia.org/wiki/Gauss-Seidel_method
    """
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process, tol, metrics = None, 
                 final_visualization = None, initial_condition = None):
        super(GaussSeidelSolver, self).__init__(self, shape, params, domains, boundary_process, tol, metrics, final_visualization)

    def solve(self, f):
        """ Solve Poissson equation with Gauss Seidel Method
        
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result after converging
        """
        # Initialization
        X = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        if self.initial_condition is not None:
            X[...] = self.initial_condition[...]

        iteration = 0
        error = 1 
        
        # Solve iteratively
        while error > self.tol:
            X = self.boundary_process(X)
                
            tmp = np.zeros_like(X) 
            tmp[self.exterior] = X[self.exterior]
            for index in range(len(self.interior[0])):
                node = (self.interior[0][index], self.interior[1][index])
                tmp[node] =  1.0 / (2.0 / self.dx / self.dx + 2.0 / self.dy / self.dy) * ((1.0 / self.dx / self.dx) * tmp[node[0] - 1, node[1]] \
                            + (1.0 / self.dx / self.dx) * X[node[0] + 1, node[1]] + (1.0 / self.dy / self.dy) * tmp[node[0], node[1] - 1] \
                            + (1.0 / self.dy / self.dy) * X[node[0], node[1] + 1] - f[node])
            
            
            error = self.metrics(tmp, X, self.interior)

            X[self.interior] = tmp[self.interior]
                
            iteration += 1
        
        # Postprocess
        print('Number of iterations in Possion Iterative Solver: ', iteration)
        
        if self.final_visualization is not None:
            self.final_visualization(X, self.dx, self.dy)
        
        return X

class SORSolver(PoissonIterativeSolver):
    """Successive over-relaxation Solver Class
        
    Implementation of successive over-relaxation Method. See reference at https://en.wikipedia.org/wiki/Successive_over-relaxation
    
    Args:
        wsor: relaxation factor
    """
    def __init__(self, shape: tuple, params: list, domains: list, boundary_process, tol, metrics = None, wsor = 1.8,
                 final_visualization = None, initial_condition = None):
        assert wsor < 2.0 and wsor > 1.0
        super(SORSolver, self).__init__(shape, params, domains, boundary_process, tol, metrics, final_visualization)
        self.wsor = wsor

    def solve(self, f):
        """ Solve Poissson equation with successive over-relaxation Method
        
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result after converging
        """
        # Initialization
        X = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        if self.initial_condition is not None:
            X[...] = self.initial_condition[...]

        iteration = 0
        error = 1 
        
        # Solve iteratively
        while error > self.tol:
            X = self.boundary_process(X)
                
            tmp = np.zeros_like(X) 
            tmp[self.exterior] = X[self.exterior]
            for index in range(len(self.interior[0])):
                node = (self.interior[0][index], self.interior[1][index])
                tmp[node] =  (1.0 - self.wsor) * X[node] + self.wsor / (2.0 / self.dx / self.dx + 2.0 / self.dy / self.dy) * ((1.0 / self.dx / self.dx) * tmp[node[0] - 1, node[1]] \
                            + (1.0 / self.dx / self.dx) * X[node[0] + 1, node[1]] + (1.0 / self.dy / self.dy) * tmp[node[0], node[1] - 1] \
                            + (1.0 / self.dy / self.dy) * X[node[0], node[1] + 1] - f[node])
            
            error = self.metrics(tmp, X, self.interior)

            X[self.interior] = tmp[self.interior]
                
            iteration += 1
        
        # Postprocess
        print('Number of iterations in Possion Iterative Solver: ', iteration)
        
        if self.final_visualization is not None:
            self.final_visualization(X, self.dx, self.dy)
        
        return X
