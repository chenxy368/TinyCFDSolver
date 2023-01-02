from solvers.poisson_iterative_solver import PointJacobiSolver, GaussSeidelSolver, SORSolver
from solvers.frac_step_solver import FracStepSolver
from utils.grid_loader import GridLoader

class FracStep():
    def __init__(self, root, metrics = None, step_visualization = None, final_visualization = None, 
                 initial_condition = None):
        assert callable(metrics) and metrics.__name__ == "<lambda>" 
        
        loader = GridLoader(root)
        
        domain_dict = {
            "u": "u_mesh.csv",
            "v": "v_mesh.csv",
            "p": "p_mesh.csv"
        }
        mesh_boundary_dict ={
            "u": ["u"],
            "v": ["v"],
            "p": ["p"]
        }

        method_info, mesh_data, _ = loader.load_grid(domain_dict, mesh_boundary_dict)
        
        self.u_boundaries = mesh_data[3]["u"]
        self.v_boundaries = mesh_data[3]["v"]
        self.p_boundaries = mesh_data[3]["p"]
        
        self.method_name = method_info[0]
        # Poisson iterative solver
        solver_name = method_info[1]

        if solver_name == "SOR":
            poisson_solver = SORSolver(mesh_data[0]["p"], 
                                       (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                       (mesh_data[2]["p"], mesh_data[2]["p_exterior"]), 
                                       self.p_boundary_process_possion_iterative, float(method_info[2]), metrics, float(method_info[3]))  
        elif solver_name == "GaussSeidel":
            poisson_solver = GaussSeidelSolver(mesh_data[0]["p"], 
                                               (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                               (mesh_data[2]["p"], mesh_data[2]["p_exterior"]), 
                                               self.p_boundary_process_possion_iterative, float(method_info[2]), metrics)
        else:
            poisson_solver = PointJacobiSolver(mesh_data[0]["p"], 
                                               (mesh_data[1]["dx"], mesh_data[1]["dy"]), 
                                               (mesh_data[2]["p"], mesh_data[2]["p_exterior"]), 
                                               self.p_boundary_process_possion_iterative, float(method_info[2]), metrics)
        
        self.solver = FracStepSolver((mesh_data[0]["u"], mesh_data[0]["v"], mesh_data[0]["p"]), 
                                     (mesh_data[1]["dt"], mesh_data[1]["dx"], mesh_data[1]["dy"], 
                                      mesh_data[1]["kinematic_viscosity"], mesh_data[1]["density"]), 
                                     (mesh_data[2]["u"], mesh_data[2]["v"], mesh_data[2]["p"], 
                                      mesh_data[2]["u_exterior"], mesh_data[2]["v_exterior"], mesh_data[2]["p_exterior"]),
                                     (self.u_boundary_process, self.v_boundary_process, self.p_boundary_process_frac_step),
                                     poisson_solver, self.extra_computing, step_visualization, final_visualization, initial_condition)

    def solve(self, num_timesteps, checkpoint_interval):
        return self.solver.solve(num_timesteps, checkpoint_interval)
    
    def u_boundary_process(self, u, v, p, t):
        for boundary in self.u_boundaries:
            u = boundary.process(u)
        return u
    
    def v_boundary_process(self, u, v, p, t):
        for boundary in self.v_boundaries:
            v = boundary.process(v)
        return v
    
    def p_boundary_process_possion_iterative(self, p):
        for boundary in self.p_boundaries:
            p = boundary.process(p)
        return p
    
    def p_boundary_process_frac_step(self, u, v, p, t):
        pass

    def extra_computing(self, u, v, p, t):
        pass