# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 04:24:08 2022

@author: Xinyang Chen
"""
import numpy as np
from solvers.poisson_iterative_solver import PointJacobiSolver, GaussSeidelSolver, SORSolver
from solvers.streamfunction_vorticity_solver import StreamFunctionVorticitySolver
from utils.grid_loader import BlendSFVGridLoader2D

class StreamFunctionVorticity():
    """streamfunction vorticity Method
    
    Implementation of streamfunction vorticity method with blended scheme for N-S equation.
    See discription and formulas at https://curiosityfluids.com/2016/03/14/streamfunction-vorticity-solution-lid-driven-cavity-flow/
    
    Attributes:
        u_boundaries: the boundary objects of u
        v_boundaries: the boundary objects of v
        psi_boundaries: the boundary objects of streamfunction only rely on streamfunction
        psiv_boundaries: the boundary objects of streamfunction only rely on streamfunction and v
        wu_boundaries: the boundary objects of vorticity only rely on vorticity and u
        wv_boundaries: the boundary objects of vorticity only rely on vorticity and v
        wuv_boundaries: the boundary objects of vorticity only rely on vorticity, u, and v
        method_name: name of the mehod, should be FracStep
        solver: a streamfunction vorticity solver
        u, v: velocity cache
    """
    def __init__(self, root, metrics = None, step_visualization = None, final_visualization = None,
                 initial_condition = None):
        assert callable(metrics) and metrics.__name__ == "<lambda>" 
        
        # Load grid
        loader = BlendSFVGridLoader2D(root)
        
        domain_dict = {
            "mesh": "mesh.csv",
        }
        mesh_boundary_dict ={
            "mesh": ["u", "v", "psi", "w_v_psi", "w_u_psi", "psi_v", "w_u_v_psi"],
        }

        method_info, mesh_data, _ = loader.load_grid(domain_dict, mesh_boundary_dict, "mesh")

        # Boundaries
        self.u_boundaries = self.check_boundary("u", mesh_data[3])
        self.v_boundaries = self.check_boundary("v", mesh_data[3])
        self.psi_boundaries = self.check_boundary("psi", mesh_data[3])
        self.psiv_boundaries = self.check_boundary("psi_v", mesh_data[3])
        self.wu_boundaries = self.check_boundary("w_u_psi", mesh_data[3])
        self.wv_boundaries = self.check_boundary("w_v_psi", mesh_data[3])
        self.wuv_boundaries = self.check_boundary("w_u_v_psi", mesh_data[3])
        
        # Name of the method
        self.method_name = method_info[0]
        
        # Poisson iterative solver
        solver_name = method_info[2]

        if solver_name == "SOR":
            poisson_solver = SORSolver(mesh_data[0]["mesh"], 
                                       (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                       (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                       self.psi_boundary_process_possion_iterative, float(method_info[3]), metrics, float(method_info[4]))  
        elif solver_name == "GaussSeidel":
            poisson_solver = GaussSeidelSolver(mesh_data[0]["mesh"], 
                                               (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                               (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                               self.psi_boundary_process_possion_iterative, float(method_info[3]), metrics)
        else:
            poisson_solver = PointJacobiSolver(mesh_data[0]["mesh"], 
                                               (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                               (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"]), 
                                               self.psi_boundary_process_possion_iterative, float(method_info[3]), metrics)

        # Inits solver
        self.solver = StreamFunctionVorticitySolver(mesh_data[0]["mesh"], 
                                     (mesh_data[1]["dt"], mesh_data[1]["dx"], mesh_data[1]["dy"], mesh_data[1]["dynamic_viscosity"]), 
                                     (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"], mesh_data[2]["wx_minus"], \
                                      mesh_data[2]["wx_plus"],  mesh_data[2]["wy_minus"], mesh_data[2]["wy_plus"]),
                                     (self.u_boundary_process, self.v_boundary_process, self.psi_boundary_process_frac_step,
                                      self.w_boundary_process),
                                     poisson_solver, float(method_info[1]),self.extra_computing, step_visualization, 
                                     final_visualization, initial_condition)
    
        self.u = np.zeros([mesh_data[0]["mesh"][0], mesh_data[0]["mesh"][1]], dtype = float)
        self.v = np.zeros([mesh_data[0]["mesh"][0], mesh_data[0]["mesh"][1]], dtype = float)

    def check_boundary(self, keyword: str, boundaries_dict: dict):
        """
        Check boundary exist or not, return empty list if no such boundary group
        """
        if keyword in boundaries_dict:
            return boundaries_dict[keyword]
        else:
            return []

    def solve(self, num_timesteps, checkpoint_interval):
        """ Call solver's solve function
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result from solver, including velocity on two direction, streamfunction and vorticity
        """
        return self.solver.solve(num_timesteps, checkpoint_interval)
    
    """
    Boundary processing functions, get variable from solver and process with the boundaries and send back
    """
    def u_boundary_process(self, u, v, psi, w, t):
        for boundary in self.u_boundaries:
            u = boundary.process(u)
        self.u[...] = u[...]
        return u
    
    def v_boundary_process(self, u, v, psi, w, t):
        for boundary in self.v_boundaries:
            v = boundary.process(v)
        self.v[...] = v[...]

        return v
    
    def psi_boundary_process_possion_iterative(self, psi):
        for boundary in self.psi_boundaries:
            psi = boundary.process(psi)
        for boundary in self.psiv_boundaries:
            psi = boundary.process(psi, (psi, self.v))
        return psi
    
    def psi_boundary_process_frac_step(self, u, v, psi, w, t):
        pass

    def w_boundary_process(self, u, v, psi, w, t):
        for boundary in self.wu_boundaries:
            w = boundary.process(w, (psi, u))
        for boundary in self.wv_boundaries:
            w = boundary.process(w, (psi, v))
        for boundary in self.wuv_boundaries:
            w = boundary.process(w, (psi, v, u))
        
        return w

    def extra_computing(self, u, v, psi, w, t):
        pass   