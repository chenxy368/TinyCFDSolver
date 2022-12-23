# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 04:24:08 2022

@author: HP
"""
import numpy as np
from methods.poisson_iterative_solver import PointJacobi, GaussSeidel, SOR

class StreamFunctionVortex():
    def __init__(self, dx: float, dy: float, dt:float, nu: float, shape: tuple, interior: list, blend_interior: list, exterior: list, 
                 metrics = None, u_boundaries = [], v_boundaries = [], psi_boundaries = [], wu_boundaries = [], wv_boundaries = [],
                 wx_plus_boundaries = [], wy_plus_boundaries = [], wx_minus_boundaries = [], wy_minus_boundaries = [], 
                 step_visualization = None, final_visualization = None, blend_factor = 0.5, solver_ID = 0, tol = 0.1, wsor = 1.8):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.nu = nu
        self.blend_factor = blend_factor
        self.shape = shape
        self.interior = interior
        self.exterior = exterior
        self.blend_interior = blend_interior
        
        self.u_boundaries = u_boundaries
        self.v_boundaries = v_boundaries
        self.psi_boundaries = psi_boundaries
        self.wu_boundaries = wu_boundaries
        self.wv_boundaries = wv_boundaries
        self.wx_plus_boundaries = wx_plus_boundaries 
        self.wy_plus_boundaries = wy_plus_boundaries 
        self.wx_minus_boundaries = wx_minus_boundaries 
        self.wy_minus_boundaries = wy_minus_boundaries
        
        # Poisson iterative solver
        solver_ID = int(solver_ID)
        if solver_ID == 0:
            self.poisson_solver = PointJacobi(self.shape, self.interior, self.exterior, 
                                              dx, dy, metrics, tol, self.psi_boundaries)
        elif solver_ID == 1:
            self.poisson_solver = GaussSeidel(self.p_shape, self.p_interior, self.p_exterior, 
                                              dx, dy, metrics, tol, self.psi_boundaries)
        else:
            self.poisson_solver = SOR(self.p_shape, self.p_interior, self.p_exterior, 
                                      dx, dy, metrics, tol, self.psi_boundaries, wsor)        

        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
    def solve(self, num_timesteps, checkpoint_interval):
        velocity_list = []
        w_list =[]
    
        # STEP 1: Initialize velocity field
        # ------------------------------------------------------
        u = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        v = np.zeros([self.shape[0], self.shape[1]], dtype = float)
    
        for boundary in self.u_boundaries:
            u = boundary.process(u)
        for boundary in self.v_boundaries:
            v = boundary.process(v)
        
        # STEP 2: Compute w on interior nodes
        # ------------------------------------------------------
        w = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        
        # ** add your code here **
        w[self.interior] = (v[self.interior[0] + 1, self.interior[1]] - v[self.interior[0] - 1, self.interior[1]]) / self.dx / 2.0 \
                           - (u[self.interior[0], self.interior[1] + 1] - u[self.interior[0], self.interior[1] - 1]) / self.dy / 2.0
        
        
        
        # STEP 3: Compute psi on interior nodes
        # ------------------------------------------------------
        psi = np.zeros([self.shape[0], self.shape[1]], dtype = float)
        
        # call iterative solver
        print('iteration number: ', 0)
        psi = self.poisson_solver.solver(-w) 
        
        # STEP 4: Compute BCs for w
        # ------------------------------------------------------
        for boundary in self.wu_boundaries:
            w = boundary.process(w, psi, u)
        for boundary in self.wv_boundaries:
            w = boundary.process(w, psi, v)
        
        # STEP 5 & 6: Solve
        # ------------------------------------------------------
        for t in range(num_timesteps):
            next_w = np.zeros_like(w) 
    
            # STEP 5: Solve Vorticity Transport Equation
            wx_minus = np.zeros_like(w) 
            wx_plus = np.zeros_like(w) 
            wy_minus = np.zeros_like(w) 
            wy_plus = np.zeros_like(w) 
            
            
            wx_minus[self.blend_interior] = (w[self.blend_interior[0] - 2, self.blend_interior[1]] - \
                                            3.0 * w[self.blend_interior[0] - 1, self.blend_interior[1]] + \
                                            3.0 * w[self.blend_interior] - \
                                            w[self.blend_interior[0] + 1, self.blend_interior[1]]) / 3.0 / self.dx
            wx_plus[self.blend_interior] = (w[self.blend_interior[0] - 1, self.blend_interior[1]] - \
                                            3.0 * w[self.blend_interior] + \
                                            3.0 * w[self.blend_interior[0] + 1, self.blend_interior[1]] - \
                                            w[self.blend_interior[0] + 2, self.blend_interior[1]]) / 3.0 / self.dx
            wy_minus[self.blend_interior] = (w[self.blend_interior[0], self.blend_interior[1] - 2] - \
                                            3.0 * w[self.blend_interior[0], self.blend_interior[1] - 1] + \
                                            3.0 * w[self.blend_interior] - \
                                            w[self.blend_interior[0], self.blend_interior[1] + 1]) / 3.0 / self.dy
            wy_plus[self.blend_interior] = (w[self.blend_interior[0], self.blend_interior[1] - 1] - \
                                            3.0 * w[self.blend_interior] + \
                                            3.0 * w[self.blend_interior[0], self.blend_interior[1] + 1] - \
                                            w[self.blend_interior[0], self.blend_interior[1] + 2]) / 3.0 / self.dy
    
            
            for boundary in self.wx_plus_boundaries:
                wx_plus = boundary.process(wx_plus, w)
            for boundary in self.wy_plus_boundaries:
                wy_plus = boundary.process(wy_plus, w)
            for boundary in self.wx_minus_boundaries:
                wx_minus = boundary.process(wx_minus, w)
            for boundary in self.wy_minus_boundaries:
                wy_minus = boundary.process(wy_minus, w)
            


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
            psi = self.poisson_solver.solver(-w) 
       
            for boundary in self.wu_boundaries:
                w = boundary.process(w, psi, u)
            for boundary in self.wv_boundaries:
                w = boundary.process(w, psi, v)

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