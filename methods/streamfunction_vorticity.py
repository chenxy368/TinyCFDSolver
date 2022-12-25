# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 04:24:08 2022

@author: HP
"""
import numpy as np
from methods.poisson_iterative_solver import PointJacobi, GaussSeidel, SOR

class StreamFunctionVortex():
    def __init__(self, method_info, mesh_data, metrics = None, step_visualization = None, final_visualization = None):
        self.shape = mesh_data[0]
        
        self.dx = self.read_input("dx", mesh_data[1])
        self.dy = self.read_input("dy", mesh_data[1])
        self.dt = self.read_input("dt", mesh_data[1])
        self.nu = self.read_input("dynamic_viscosity", mesh_data[1])
        
        self.interior = self.read_input("mesh", mesh_data[2])
        self.exterior = self.read_input("mesh_exterior", mesh_data[2])
        self.blend_interior = self.read_input("blend_mesh", mesh_data[2])

        self.u_boundaries = self.read_input("u", mesh_data[3])
        self.v_boundaries = self.read_input("v", mesh_data[3])
        self.psi_boundaries = self.read_input("psi", mesh_data[3])
        self.wu_boundaries = self.read_input("w_u_psi", mesh_data[3])
        self.wv_boundaries = self.read_input("w_v_psi", mesh_data[3])
        self.wx_plus_boundaries = self.read_input("wx_plus", mesh_data[3])
        self.wy_plus_boundaries = self.read_input("wy_plus", mesh_data[3])
        self.wx_minus_boundaries = self.read_input("wx_minus", mesh_data[3])
        self.wy_minus_boundaries = self.read_input("wy_minus", mesh_data[3])
        
        self.method_name = method_info[0]
        self.blend_factor = float(method_info[1])
        # Poisson iterative solver
        solver_name = method_info[2]
        poisson_solver_domain_dict = {
            "domain": self.interior,
            "domain_exterior": self.exterior
        }
        poisson_solver_boundary_dict = {
            "boundary": self.psi_boundaries
        }
        poisson_method_info = method_info[2:]

        poisson_mesh_data = (mesh_data[0], mesh_data[1], poisson_solver_domain_dict, poisson_solver_boundary_dict)
        if solver_name == "SOR":
            self.poisson_solver = SOR(poisson_method_info, poisson_mesh_data, metrics)  
        elif solver_name == "GaussSeidel":
            self.poisson_solver = GaussSeidel(poisson_method_info, poisson_mesh_data, metrics)
        else:
            self.poisson_solver = PointJacobi(poisson_method_info, poisson_mesh_data, metrics)

        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
    
    def read_input(self, keyword: str, input_dict: dict):
        if keyword in input_dict:
            return input_dict[keyword]
        else:
            raise RuntimeError("MISSING INFORMATION")
    
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