# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 04:24:08 2022

@author: HP
"""
from solvers.poisson_iterative_solver import PointJacobiSolver, GaussSeidelSolver, SORSolver
from solvers.streamfunction_vorticity_solver import StreamFunctionVorticitySolver
from utils.grid_loader import BlendSFVGridLoader2D

class StreamFunctionVorticity():
    def __init__(self, root, metrics = None, step_visualization = None, final_visualization = None):
        assert callable(metrics) and metrics.__name__ == "<lambda>" 
        
        loader = BlendSFVGridLoader2D(root)
        
        domain_dict = {
            "mesh": "mesh.csv",
        }
        mesh_boundary_dict ={
            "mesh": ["u", "v", "psi", "w_v_psi", "w_u_psi"],
        }

        method_info, mesh_data = loader.load_grid(domain_dict, mesh_boundary_dict, "mesh")

        self.u_boundaries = mesh_data[3]["u"]
        self.v_boundaries = mesh_data[3]["v"]
        self.psi_boundaries = mesh_data[3]["psi"]
        self.wu_boundaries = mesh_data[3]["w_u_psi"]
        self.wv_boundaries = mesh_data[3]["w_v_psi"]
        
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

        self.solver = StreamFunctionVorticitySolver(mesh_data[0]["mesh"], 
                                     (mesh_data[1]["dt"], mesh_data[1]["dx"], mesh_data[1]["dy"], mesh_data[1]["dynamic_viscosity"]), 
                                     (mesh_data[2]["mesh"], mesh_data[2]["mesh_exterior"], mesh_data[2]["wx_minus"], \
                                      mesh_data[2]["wx_plus"],  mesh_data[2]["wy_minus"], mesh_data[2]["wy_plus"]),
                                     (self.u_boundary_process, self.v_boundary_process, self.psi_boundary_process_frac_step,
                                      self.w_boundary_process),
                                     poisson_solver, float(method_info[1]),self.extra_computing, step_visualization, final_visualization)
    

    def solve(self, num_timesteps, checkpoint_interval):
        return self.solver.solve(num_timesteps, checkpoint_interval)
    
    def u_boundary_process(self, u, v, psi, w, t):
        for boundary in self.u_boundaries:
            u = boundary.process(u)
        return u
    
    def v_boundary_process(self, u, v, psi, w, t):
        for boundary in self.v_boundaries:
            v = boundary.process(v)
        return v
    
    def psi_boundary_process_possion_iterative(self, psi):
        for boundary in self.psi_boundaries:
            psi = boundary.process(psi)
        return psi
    
    def psi_boundary_process_frac_step(self, u, v, psi, w, t):
        pass

    def w_boundary_process(self, u, v, psi, w, t):
        for boundary in self.wu_boundaries:
            w = boundary.process(w, (psi, u))
        for boundary in self.wv_boundaries:
            w = boundary.process(w, (psi, v))
        
        return w

    def extra_computing(self, u, v, psi, w, t):
        pass   