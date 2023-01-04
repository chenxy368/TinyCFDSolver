# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 03:12:39 2022

@author: Xinyang Chen
"""
import numpy as np

class FracStepSolver():
    """Fractional Step Solver
    
    Implementation of fractional step method for N-S equation. Solver is called in method.
    See discription and formulas at https://en.wikipedia.org/wiki/Projection_method_(fluid_dynamics)
    
    Attributes:
        u_shape: the shape of u's grid
        v_shape: the shape of v's grid
        p_shape: the shape of p's grid
        dt, dx, dy, mu, rho: timestep length, grid length, kinematic_viscosity, density
        u_interior, v_interior, p_interior: interior slice of u, v and p's grid
        u_exterior, v_exterior, p_exterior: exterior slice of u, v and p's grid
        u_boundary_process, v_boundary_process, p_boundary_process: boundaries process functions of u, v and p
        poisson_solver: a poisson iterative solver
        extra_computing: extra computing at the end of each timestep
        step_visualization: visualization function at each timestep
        final_visualization: final visualization function
        initial_condition: initial conditions of u, v and p
    """
    def __init__(self, shapes: tuple, params: list, domains: list, boundaries: list, poisson_solver, extra_computing = None,
                 step_visualization = None, final_visualization = None, initial_condition = None):
        # Check Initial Condition's Shape
        if initial_condition is not None and (len(initial_condition) != 3 or initial_condition[0].shape != shapes[0] \
           or initial_condition[1].shape != shapes[1] or initial_condition[2].shape != shapes[2]):
            raise RuntimeError("Initial Condition Shape Dismatch")
        
        # Domain shapes
        self.u_shape = shapes[0]
        self.v_shape = shapes[1]
        self.p_shape = shapes[2]

        # Simulation parameters
        self.dt = params[0]
        self.dx = params[1]
        self.dy = params[2]
        self.mu = params[3]
        self.rho = params[4]

        # Interior and exterior
        self.u_interior = domains[0]
        self.v_interior = domains[1]
        self.p_interior = domains[2]
        self.u_exterior = domains[3]
        self.v_exterior = domains[4]
        self.p_exterior = domains[5]
        
        # Boundary process
        self.u_boundary_process = boundaries[0]
        self.v_boundary_process = boundaries[1]
        self.p_boundary_process = boundaries[2]
        
        # Poisson iterative solver
        self.poisson_solver = poisson_solver
        
        # Post processor
        self.extra_computing = extra_computing
        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
        
        # Initial Condition
        self.initial_condition = initial_condition
        
    def solve(self, num_timesteps, checkpoint_interval):
        """ Solve N-S equation with fraction step method
       
        1. Initialization
        2. Impose boundary
        3. Predict velocity
        4. Solve pressure
        5. Correct velocity
        6. Repeat 2-5 till end
        
        Args:
            num_timesteps: the number of total timesteps
            checkpoint_interval: frequency of calling step postprocess
        Return:
            result tuple (u, v, p)
        """
        # Result List
        velocity_list = []
        pressure_list = []
        
        # Initialization
        u = np.zeros([self.u_shape[0], self.u_shape[1]], dtype = float)
        v = np.zeros([self.v_shape[0], self.v_shape[1]], dtype = float)
        p = np.zeros([self.p_shape[0], self.p_shape[1]], dtype = float)
        
        if self.initial_condition is not None:
            u[...] = self.initial_condition[0][...]
            v[...] = self.initial_condition[1][...]
            p[...] = self.initial_condition[2][...]
        
        # Solve
        for t in range(num_timesteps):
            print('Timestep: ', t)
            
            # Step 1: Impose boundary
            u = self.u_boundary_process(u, v, p, t)
            v = self.v_boundary_process(u, v, p, t)
            
            # Step 2: predict u_star and v_star
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


            u_star = self.u_boundary_process(u_star, v_star, p, t)
            v_star = self.v_boundary_process(u_star, v_star, p, t)    
            self.p_boundary_process = (u_star, v_star, p, t)
            
            # Step 3: solve p
            # construct the right hand side of the pressure equation
            rhs_p = np.zeros([self.p_shape[0], self.p_shape[1]], dtype = float)
            rhs_p[self.p_interior] = self.rho / self.dt * ((u_star[self.p_interior] - u_star[(self.p_interior[0] - 1, self.p_interior[1])]) / self.dx \
                + (v_star[(self.p_interior[0], self.p_interior[1])] - v_star[(self.p_interior[0], self.p_interior[1] - 1)]) / self.dy)
            
            # call iterative solver
            p = self.poisson_solver.solve(rhs_p)
            
            # Step 4: correct u_star and v_star
            u[self.u_interior] = u_star[self.u_interior] - self.dt / self.rho / self.dx * (p[(self.u_interior[0] + 1, self.u_interior[1])] - p[self.u_interior])
            v[self.v_interior] = v_star[self.v_interior] - self.dt / self.rho / self.dy * (p[(self.v_interior[0], self.v_interior[1] + 1)] - p[self.v_interior])
            
            # Extra computing in each timestep
            self.extra_computing(u, v, p, t)
            
            # Checkpoint
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
        
        # Final postprocess
        if self.final_visualization is not None:
            self.final_visualization(velocity_list, pressure_list, self.dx, self.dy)

        return (u, v, p)