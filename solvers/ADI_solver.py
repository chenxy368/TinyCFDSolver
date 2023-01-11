# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 08:18:01 2023

@author: Xinyang Chen
"""
import numpy as np
from numba import jit

class ADISolver():
    """Alternating-direction implicit method Solver
    
    Implementation of alternating-direction implicit method method for diffusion equation. Solver is called in method.
    See discription and formulas at https://en.wikipedia.org/wiki/Alternating-direction_implicit_method
    
    Attributes:
        shape: the shape of grid
        problem_set_x, problem_set_y: 1D problem sets along two directions
        dt, dx, dy, alpha: timestep length, grid length, diffusion coefficient 
        interior: interior slice of grid
        exterior: exterior slice of grid
        boundary_process_2D: boundaries process functions working on 2D domain
        boundary_process_RHS: boundaries process functions working on right hand side
        use_TDMA: use TDMA or not
        extra_computing: extra computing at the end of each timestep
        step_visualization: visualization function at each timestep
        final_visualization: final visualization function
        initial_condition: initial conditions
    """
    def __init__(self, shape: tuple, problem_set_x: list, problem_set_y: list, params: list, domains: list, 
                 boundary_process_2D, boundary_process_RHS, use_TDMA = False, extra_computing = None, 
                 step_visualization = None, final_visualization = None, initial_condition = None):
        self.shape = shape
        self.problem_set_x = problem_set_x
        self.problem_set_y = problem_set_y
        
        self.dx = params[0]
        self.dy = params[1]
        self.dt = params[2]
        self.alpha = params[3]

        self.interior = domains[0]
        self.exterior = domains[1]    

        self.boundary_process_2D = boundary_process_2D
        self.boundary_process_RHS = boundary_process_RHS
        self.use_TDMA = use_TDMA
        self.extra_computing = extra_computing

        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
        
        self.initial_condition = initial_condition
    
    def solve(self, num_timesteps, checkpoint_interval):
        """ Solve advection diffusion equation with upwind central scheme method
        
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result X
        """
        betax = self.alpha * self.dt / (self.dx * self.dx)
        betay = self.alpha * self.dt / (self.dy * self.dy)
        
        X_list = []
        # Initialize
        X = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        if self.initial_condition is not None:
            X[...] = self.initial_condition[...]  

        for t in range(num_timesteps):
            print("Timestep:" + str(t))
            # STEP 1
            # Impose boundary conditions
            X = self.boundary_process_2D(X, t)

            # STEP 2
            # Solve x direction
            X_tmp = np.zeros_like(X, dtype = float)
            for curr_set in self.problem_set_x:
                # Get interior and position
                curr_interior = curr_set[2]
                y = curr_set[3]
                # Define vector b
                b = betay * X[curr_interior, y-1] + (1 - 2 * betay) * X[curr_interior, y] + betay * X[curr_interior, y+1]
                b = b.transpose()
                b = self.boundary_process_RHS(b, t, curr_set[0])
                
                # Solve Ax = b
                if self.use_TDMA:
                    sub = np.zeros_like(curr_set[1][0])
                    dia = np.zeros_like(curr_set[1][1])
                    sup = np.zeros_like(curr_set[1][2])
                    res = np.zeros_like(b.reshape(-1))
                    sub[...] = curr_set[1][0][...]
                    dia[...] = curr_set[1][1][...]
                    sup[...] = curr_set[1][2][...]
                    res[...] = b[..., 0]
                    b = self.TDMA(sub, dia, sup, res)
                else:
                    b = np.linalg.solve(curr_set[1], b).reshape(-1)
                
                # Assign
                X_tmp[curr_interior, y] = b

            X[self.interior] = X_tmp[self.interior]
            
            X_tmp = np.zeros_like(X, dtype = float)

            # STEP 3
            # Solve y direction
            for curr_set in self.problem_set_y:
                # Get interior and position
                curr_interior = curr_set[2]
                x = curr_set[3]
                
                # Define vector b
                b = betax * X[x-1, curr_interior] + (1 - 2 * betax) * X[x, curr_interior] + betax * X[x+1, curr_interior]
                b = b.transpose()
                b = self.boundary_process_RHS(b, t, curr_set[0])
                
                # Solve Ax = b
                if self.use_TDMA:
                    sub = np.zeros_like(curr_set[1][0])
                    dia = np.zeros_like(curr_set[1][1])
                    sup = np.zeros_like(curr_set[1][2])
                    res = np.zeros_like(b.reshape(-1))
                    sub[...] = curr_set[1][0][...]
                    dia[...] = curr_set[1][1][...]
                    sup[...] = curr_set[1][2][...]
                    res[...] = b[..., 0]
                    b = self.TDMA(sub, dia, sup, res)
                else:
                    b = np.linalg.solve(curr_set[1], b).reshape(-1)

                # Assgin
                X_tmp[x, curr_interior] = b

            X[self.interior] = X_tmp[self.interior]   
           
            self.extra_computing(X, t)
            if self.final_visualization is not None and (t + 1) % checkpoint_interval == 0:
                X_list.append(np.transpose(X).copy())
                
                if self.step_visualization is not None:
                    self.step_visualization(X, self.dx, self.dy, self.dt, t)

        if self.final_visualization is not None:
            self.final_visualization(X_list, self.dx, self.dy)

        return X
    
    @staticmethod
    @jit
    def TDMA(a,b,c,d):
        """ Tridiagonal matrix algorithm
        
        In numerical linear algebra, the tridiagonal matrix algorithm, also known as the Thomas algorithm 
        (named after Llewellyn Thomas), is a simplified form of Gaussian elimination that can be used to 
        solve tridiagonal systems of equations. See more information at 
        https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        
        Args:
            a: subdiagonal of A
            b: diagonal of A
            c: superdiagonal of A
            d: RHS vector
        Return:
            result x
        """


        N = len(d)
        for i in range(N-1):
            w = a[i + 1] / b[i]
            b[i + 1] -= w * c[i]
            d[i + 1] -= w * d[i]
 
        x = b
        x[N-1] = d[N-1] / b[N-1]
        for i in range(N - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i+1]) / b[i]
        
        return x

