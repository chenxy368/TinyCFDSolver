import numpy as np

class FracStepSolver():
    def __init__(self, mu, rho, dt, dx, dy, u_shape, v_shape, p_shape, 
                 u_boundaries, v_boundaries, p_boundaries, u_interior, v_interior, p_interior,
                 u_exterior, v_exterior, p_exterior, step_visualization = None, final_visualization = None, tol = 0.1):
        # define constants
        self.mu = mu
        self.rho = rho

        # Simulation parameters
        self.dt = dt
        self.dx = dx
        self.dy = dy
    
        # Domain information
        self.u_shape = u_shape
        self.v_shape = v_shape
        self.p_shape = p_shape
        
        self.u_interior = u_interior
        self.v_interior = v_interior
        self.p_interior = p_interior
        self.u_exterior = u_exterior
        self.v_exterior = v_exterior 
        self.p_exterior = p_exterior 
        
        # Boundary information
        self.u_boundaries = u_boundaries
        self.v_boundaries = v_boundaries
        self.p_boundaries = p_boundaries
        
        # Tolerance of Poisson iterative solver
        self.tol = tol
        
        # Post processor
        self.step_visualization = step_visualization
        self.final_visualization = final_visualization
        
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
            p = self.poisson_iterative_solver(rhs_p)

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
   
    def poisson_iterative_solver(self, f):
        X = np.zeros([self.p_shape[0], self.p_shape[1]], dtype = float)
        # initialize the errors
        error = 1
        iteration = 0

        # iterate until the error is less than the tolerance or the maximum number of iterations is reached
        # while error > tol and iter < maxiter:
        while error > self.tol:
            for boundary in self.p_boundaries:
                X = boundary.process(X)
        
            X_ref = X.copy()
            
            # update the interior points
            X[self.p_interior] = 1 / 4 * (X[self.p_interior[0] + 1, self.p_interior[1]] + X[self.p_interior[0] - 1, self.p_interior[1]] + X[self.p_interior[0], self.p_interior[1] + 1] + X[self.p_interior[0], self.p_interior[1] - 1]) - \
                self.dx ** 2 / 4 * f[self.p_interior]

            # calculate the error
            error = np.max(np.abs(X[self.p_interior] - X_ref[self.p_interior]))
        
            # update the iteration
            iteration += 1
        
        print('Number of iterations: ', iteration)

        return X