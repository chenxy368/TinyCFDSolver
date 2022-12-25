import numpy as np
from methods.poisson_iterative_solver import PointJacobi, GaussSeidel, SOR

class FracStepSolver():
    def __init__(self, method_info, mesh_data, metrics = None, step_visualization = None, final_visualization = None):
        assert callable(step_visualization) and step_visualization.__name__ == "<lambda>" and callable(final_visualization) \
            and final_visualization.__name__ == "<lambda>" and callable(metrics) and metrics.__name__ == "<lambda>" 
         
        # Domain information
        self.u_shape = mesh_data[0][0]
        self.v_shape = mesh_data[0][1]
        self.p_shape = mesh_data[0][2]

        # Simulation parameters
        self.dt = self.read_input("dt", mesh_data[1])
        self.dx = self.read_input("dx", mesh_data[1])
        self.dy = self.read_input("dy", mesh_data[1])
        
        # define constants
        self.mu = self.read_input("kinematic_viscosity", mesh_data[1])
        self.rho = self.read_input("density", mesh_data[1])

        # Domain information
        self.u_interior = self.read_input("u", mesh_data[2])
        self.v_interior = self.read_input("v", mesh_data[2])
        self.p_interior = self.read_input("p", mesh_data[2])
        self.u_exterior = self.read_input("u_exterior", mesh_data[2])
        self.v_exterior = self.read_input("v_exterior", mesh_data[2]) 
        self.p_exterior = self.read_input("p_exterior", mesh_data[2])  
        
        # Boundary information
        self.u_boundaries = self.read_input("u", mesh_data[3])
        self.v_boundaries = self.read_input("v", mesh_data[3])
        self.p_boundaries = self.read_input("p", mesh_data[3])
        
        self.method_name = method_info[0]
        # Poisson iterative solver
        solver_name = method_info[1]
        poisson_solver_domain_dict = {
            "domain": self.p_interior,
            "domain_exterior": self.p_exterior
        }
        poisson_solver_boundary_dict = {
            "boundary": self.p_boundaries
        }
        poisson_method_info = method_info[1:]

        poisson_mesh_data = (self.p_shape, mesh_data[1], poisson_solver_domain_dict, poisson_solver_boundary_dict)
        if solver_name == "SOR":
            self.poisson_solver = SOR(poisson_method_info, poisson_mesh_data, metrics)  
        elif solver_name == "GaussSeidel":
            self.poisson_solver = GaussSeidel(poisson_method_info, poisson_mesh_data, metrics)
        else:
            self.poisson_solver = PointJacobi(poisson_method_info, poisson_mesh_data, metrics)
        # Post processor
        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
        
    def read_input(self, keyword: str, input_dict: dict):
        if keyword in input_dict:
            return input_dict[keyword]
        else:
            raise RuntimeError("MISSING INFORMATION")
        
    def solve(self, num_timesteps, checkpoint_interval):
        # animation
        velocity_list = []
        pressure_list = []
        
        u = np.zeros([self.u_shape[0], self.u_shape[1]], dtype = float)
        v = np.zeros([self.v_shape[0], self.v_shape[1]], dtype = float)
        p = np.zeros([self.p_shape[0], self.p_shape[1]], dtype = float)
        
        # time loop
        for t in range(num_timesteps):
            for boundary in self.u_boundaries:
                u = boundary.process(u)
            for boundary in self.v_boundaries:
                v = boundary.process(v)
            
            # Step 1: predict u_star and v_star
            u_star = u.copy()
            u_a = u.copy()
            u_b = u.copy()
            v_star = v.copy()
            v_a = v.copy()
            v_b = v.copy()

            # u_a: advection term
            u_a[self.u_interior] = u_a[self.u_interior] = -1.0 / self.dx / 4.0 * ((u[(self.u_interior[0] + 1, self.u_interior[1])] + u[self.u_interior]) ** 2 \
                - (u[self.u_interior] + u[(self.u_interior[0] - 1, self.u_interior[1])]) ** 2) \
                -1.0 / self.dx / 4.0 * ((u[(self.u_interior[0], self.u_interior[1] + 1)] + u[self.u_interior]) * (v[(self.u_interior[0] + 1, self.u_interior[1])] + v[self.u_interior]) \
                - (u[self.u_interior] + u[(self.u_interior[0], self.u_interior[1] - 1)]) * (v[(self.u_interior[0] + 1, self.u_interior[1] - 1)] + v[(self.u_interior[0], self.u_interior[1] - 1)]))
            
            # u_b: diffusion term
            u_b[self.u_interior] = self.mu / self.dx ** 2 * (u[(self.u_interior[0] + 1, self.u_interior[1])] - 2 * u[self.u_interior] + u[(self.u_interior[0] - 1, self.u_interior[1])]) \
                + self.mu / self.dy ** 2 * (u[(self.u_interior[0], self.u_interior[1] + 1)] - 2 * u[self.u_interior] + u[(self.u_interior[0], self.u_interior[1] - 1)])

            # v_a: advection term
            v_a[self.v_interior] = -1.0 / self.dy / 4.0 * ((v[(self.v_interior[0], self.v_interior[1] + 1)] + v[self.v_interior]) ** 2 \
                - (v[self.v_interior] + v[(self.v_interior[0], self.v_interior[1] - 1)]) ** 2) \
                -1.0 / self.dy / 4.0 * ((v[(self.v_interior[0] + 1, self.v_interior[1])] + v[self.v_interior]) * (u[(self.v_interior[0], self.v_interior[1] + 1)] + u[self.v_interior]) \
                - (v[self.v_interior] + v[(self.v_interior[0] - 1, self.v_interior[1])]) * (u[(self.v_interior[0] - 1, self.v_interior[1] + 1)] + u[(self.v_interior[0] - 1, self.v_interior[1])]))

            # v_b: diffusion term
            v_b[self.v_interior] = self.mu / self.dx ** 2 * (v[(self.v_interior[0] + 1, self.v_interior[1])] - 2 * v[self.v_interior] + v[(self.v_interior[0] - 1, self.v_interior[1])]) \
                + self.mu / self.dy ** 2 * (v[(self.v_interior[0], self.v_interior[1] + 1)] - 2 * v[self.v_interior] + v[(self.v_interior[0], self.v_interior[1] - 1)])
            
            # update u_star and v_star
            u_star[self.u_interior] = u[self.u_interior] + self.dt * (u_a[self.u_interior] + u_b[self.u_interior])
            v_star[self.v_interior] = v[self.v_interior] + self.dt * (v_a[self.v_interior] + v_b[self.v_interior])


            for boundary in self.u_boundaries:
                u_star = boundary.process(u_star)
            for boundary in self.v_boundaries:
                v_star = boundary.process(v_star)
                
            # Step 2: solve for p
            # construct the right hand side of the pressure equation
            rhs_p = np.zeros([self.p_shape[0], self.p_shape[1]], dtype = float)
            rhs_p[self.p_interior] = self.rho / self.dt * ((u_star[self.p_interior] - u_star[(self.p_interior[0] - 1, self.p_interior[1])]) / self.dx \
                + (v_star[(self.p_interior[0], self.p_interior[1])] - v_star[(self.p_interior[0], self.p_interior[1] - 1)]) / self.dy)
            
            # call iterative solver
            print('iteration number: ', t)
            p = self.poisson_solver.solver(rhs_p)
            
            # Step 3: correct u_star and v_star
            u[self.u_interior] = u_star[self.u_interior] - self.dt / self.rho / self.dx * (p[(self.u_interior[0] + 1, self.u_interior[1])] - p[self.u_interior])
            v[self.v_interior] = v_star[self.v_interior] - self.dt / self.rho / self.dy * (p[(self.v_interior[0], self.v_interior[1] + 1)] - p[self.v_interior])
            if self.final_visualization is not None and (t + 1) % checkpoint_interval == 0:
                u[self.u_exterior] = 0
                v[self.v_exterior] = 0
                p[self.p_exterior] = 0
                
                u_comp = np.zeros_like((self.p_shape[0]-2, self.p_shape[1]-2), dtype = float)
                v_comp = np.zeros_like((self.p_shape[0]-2, self.p_shape[1]-2), dtype = float)
                u_comp = (u[0:-1, 1:-1] + u[1:, 1:-1]) / 2
                v_comp = (v[1:-1, 0:-1] + v[1:-1, 1:]) / 2


                velocity = np.sqrt(u_comp ** 2 + v_comp ** 2)
                velocity = velocity.transpose()
                velocity_list.append(velocity)
                pressure_list.append((np.transpose(p)))
                
                if self.step_visualization is not None:
                    self.step_visualization(u_comp, v_comp, velocity, p, self.dx, self.dy, self.dt, t)

        if self.final_visualization is not None:
            self.final_visualization(velocity_list, pressure_list, self.dx, self.dy)
