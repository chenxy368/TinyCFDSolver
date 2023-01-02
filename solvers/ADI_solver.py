# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 08:18:01 2023

@author: HP
"""
import numpy as np

class ADISolver():
    def __init__(self, shape: tuple, problem_set_x: list, problem_set_y: list, params: list, domains: list, 
                 boundary_process_2D, boundary_process_RHS, extra_computing = None, 
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
        self.extra_computing = extra_computing

        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
        
        self.initial_condition = initial_condition
    
    def solve(self, num_timesteps, checkpoint_interval):
        betax = self.alpha * self.dt / (self.dx * self.dx)
        betay = self.alpha * self.dt / (self.dy * self.dy)
        
        X_list = []
        # Initialize
        X = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        X[...] = 300
        # Initial conditions
        if self.initial_condition is not None:
            X[...] = self.initial_condition[...]  
            
        for t in range(num_timesteps):
            # STEP 1
            # Impose boundary conditions
            X = self.boundary_process_2D(X, t)
        
            X_tmp = np.zeros_like(X, dtype = float)
            X_tmp = self.boundary_process_2D(X_tmp, t)
            
            for curr_set in self.problem_set_x:
                curr_interior = curr_set[2]
                y = curr_set[3]
                # Define vector b
                b = betay * X[curr_interior, y-1] + (1 - 2 * betay) * X[curr_interior, y] + betay * X[curr_interior, y+1]
                b = b.transpose()
                b = self.boundary_process_RHS(b, t, curr_set[0])
                
                b = np.linalg.solve(curr_set[1], b).reshape(-1)

                '''
                elif solverID == 2:
                    subdiagonal = np.full((Nx-1), -betax, dtype = float)
                    subdiagonal[0] = 0
                    diagonal = np.full((Nx-1), 1 + 2 * betax, dtype = float)
                    superdiagonal = np.full((Nx-1), -betax, dtype = float)
                    superdiagonal[Nx-2] = 0
                    b = TDMA(subdiagonal, diagonal, superdiagonal, b)
                '''
                X_tmp[curr_interior, y] = b

            X[self.interior] = X_tmp[self.interior]
            
            # STEP 2
            # Impose boundary conditions
            X = self.boundary_process_2D(X, t)
        
            X_tmp = np.zeros_like(X, dtype = float)
            X_tmp = self.boundary_process_2D(X_tmp, t)
            
            for curr_set in self.problem_set_y:
                curr_interior = curr_set[2]
                x = curr_set[3]
                # Define vector b
                b = betax * X[x-1, curr_interior] + (1 - 2 * betax) * X[x, curr_interior] + betax * X[x+1, curr_interior]
                b = b.transpose()
                b = self.boundary_process_RHS(b, t, curr_set[0])
    
                # Solve Ax=b
                b = np.linalg.solve(curr_set[1], b).reshape(-1)
                '''
                elif solverID == 2:
                    subdiagonal = np.full((Ny-1), -betay, dtype = float)
                    subdiagonal[0] = 0
                    diagonal = np.full((Ny-1), 1 + 2 * betay, dtype = float)
                    superdiagonal = np.full((Ny-1), -betay, dtype = float)
                    superdiagonal[Ny-2] = 0
                    b = TDMA(subdiagonal, diagonal, superdiagonal, b)
                '''
                X_tmp[x, curr_interior] = b

            X[self.interior] = X_tmp[self.interior]   

            self.extra_computing(X, t)
            if self.final_visualization is not None and (t + 1) % checkpoint_interval == 0:
                X_list.append((np.transpose(X)))
                
                if self.step_visualization is not None:
                    self.step_visualization(X, self.dx, self.dy, self.dt, t)

        if self.final_visualization is not None:
            self.final_visualization(X_list, self.dx, self.dy)

        return X