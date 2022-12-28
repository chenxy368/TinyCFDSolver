# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 04:24:08 2022

@author: HP
"""
import numpy as np

class StreamFunctionVorticitySolver():
    def __init__(self, shape: tuple, params: list, domians: list, boundaries: list, poisson_solver, blend_factor = 0, extra_computing = None,
                 step_visualization = None, final_visualization = None):
        self.shape = shape
        
        self.dt = params[0]
        self.dx = params[1]
        self.dy = params[2]
        self.nu = params[3]
        
        self.interior = domians[0]
        self.exterior = domians[1]
        
        self.wx_minus_interior = domians[2]
        self.wx_plus_interior = domians[3]
        self.wy_minus_interior = domians[4]
        self.wy_plus_interior = domians[5]

        self.u_boundary_process = boundaries[0]
        self.v_boundary_process = boundaries[1]
        self.psi_boundary_process = boundaries[2]
        self.w_boundary_process = boundaries[3]
        
        self.poisson_solver = poisson_solver
        self.blend_factor = blend_factor
        
        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
    
    def solve(self, num_timesteps, checkpoint_interval):
        velocity_list = []
        w_list =[]
    
        # STEP 1: Initialize velocity field
        # ------------------------------------------------------
        u = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        v = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        w = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        psi = np.zeros([self.shape[0], self.shape[1]], dtype = float)

        u = self.u_boundary_process(u, v, psi, w, 0)
        v = self.v_boundary_process(u, v, psi, w, 0)
        
        # STEP 2: Compute w on interior nodes
        # ------------------------------------------------------
        w[self.interior] = (v[self.interior[0] + 1, self.interior[1]] - v[self.interior[0] - 1, self.interior[1]]) / self.dx / 2.0 \
                           - (u[self.interior[0], self.interior[1] + 1] - u[self.interior[0], self.interior[1] - 1]) / self.dy / 2.0
        
        # STEP 3: Compute psi on interior nodes
        # ------------------------------------------------------
        # call iterative solver
        print('iteration number: ', 0)
        psi = self.poisson_solver.solve(-w) 
        
        # STEP 4: Compute BCs for w
        # ------------------------------------------------------
        w = self.w_boundary_process(u, v, psi, w, 0)
        
        # STEP 5 & 6: Solve
        # ------------------------------------------------------
        for t in range(num_timesteps):
            u = self.u_boundary_process(u, v, psi, w, t)
            v = self.v_boundary_process(u, v, psi, w, t)

            next_w = np.zeros_like(w) 
    
            # STEP 5: Solve Vorticity Transport Equation
            wx_minus = np.zeros_like(w) 
            wx_plus = np.zeros_like(w) 
            wy_minus = np.zeros_like(w) 
            wy_plus = np.zeros_like(w) 

            wx_minus[self.wx_minus_interior] = (w[self.wx_minus_interior[0] - 2, self.wx_minus_interior[1]] - \
                                            3.0 * w[self.wx_minus_interior[0] - 1, self.wx_minus_interior[1]] + \
                                            3.0 * w[self.wx_minus_interior] - \
                                            w[self.wx_minus_interior[0] + 1, self.wx_minus_interior[1]]) / 3.0 / self.dx
            wx_plus[self.wx_plus_interior] = (w[self.wx_plus_interior[0] - 1, self.wx_plus_interior[1]] - \
                                            3.0 * w[self.wx_plus_interior] + \
                                            3.0 * w[self.wx_plus_interior[0] + 1, self.wx_plus_interior[1]] - \
                                            w[self.wx_plus_interior[0] + 2, self.wx_plus_interior[1]]) / 3.0 / self.dx
            wy_minus[self.wy_minus_interior] = (w[self.wy_minus_interior[0], self.wy_minus_interior[1] - 2] - \
                                            3.0 * w[self.wy_minus_interior[0], self.wy_minus_interior[1] - 1] + \
                                            3.0 * w[self.wy_minus_interior] - \
                                            w[self.wy_minus_interior[0], self.wy_minus_interior[1] + 1]) / 3.0 / self.dy
            wy_plus[self.wy_plus_interior] = (w[self.wy_plus_interior[0], self.wy_plus_interior[1] - 1] - \
                                            3.0 * w[self.wy_plus_interior] + \
                                            3.0 * w[self.wy_plus_interior[0], self.wy_plus_interior[1] + 1] - \
                                            w[self.wy_plus_interior[0], self.wy_plus_interior[1] + 2]) / 3.0 / self.dy

            next_w[self.interior] = (1.0 - 2.0 * self.nu * self.dt / self.dx / self.dx - 2.0 * self.nu * self.dt / self.dy / self.dy) * w[self.interior] \
                                + (self.nu * self.dt / self.dx / self.dx - self.dt / 2.0 / self.dx * u[self.interior]) * w[self.interior[0] + 1, self.interior[1]] \
                                + (self.nu * self.dt / self.dx / self.dx + self.dt / 2.0 / self.dx * u[self.interior]) * w[self.interior[0] - 1, self.interior[1]] \
                                + (self.nu * self.dt / self.dy / self.dy - self.dt / 2.0 / self.dy * v[self.interior]) * w[self.interior[0], self.interior[1] + 1] \
                                + (self.nu * self.dt / self.dy / self.dy + self.dt / 2.0 / self.dy * v[self.interior]) * w[self.interior[0], self.interior[1] - 1] \
                                - self.blend_factor * self.dt * (np.maximum(u[self.interior], next_w[self.interior]) * wx_minus[self.interior] \
                                + np.minimum(u[self.interior], next_w[self.interior]) * wx_plus[self.interior] \
                                + np.maximum(v[self.interior], next_w[self.interior]) * wy_minus[self.interior] \
                                + np.minimum(v[self.interior], next_w[self.interior]) * wy_plus[self.interior])
                                
            # STEP 6: Solve Poisson Equation
            w[self.interior] = next_w[self.interior]

            print('iteration number: ', t)
            psi = self.poisson_solver.solve(-w) 

            w = self.w_boundary_process(u, v, psi, w, t)
            
            u[self.interior] = (psi[self.interior[0], self.interior[1] + 1] - psi[self.interior[0], self.interior[1] - 1]) / self.dy / 2.0 
            v[self.interior] = -(psi[self.interior[0] + 1, self.interior[1]] - psi[self.interior[0] - 1, self.interior[1]]) / self.dx / 2.0


            if self.final_visualization is not None and (t + 1) % checkpoint_interval == 0:
                velocity = np.sqrt(u ** 2 + v ** 2)
                velocity = velocity.transpose()
                velocity_list.append(velocity)
                w_list.append((np.transpose(w)))
                
                if self.step_visualization is not None:
                    self.step_visualization(u, v, velocity, w, self.dx, self.dy, self.dt, t)

        if self.final_visualization is not None:
            self.final_visualization(velocity_list, w_list, self.dx, self.dy)