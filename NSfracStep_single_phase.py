import numpy as np
import matplotlib.pyplot as plt
from solvers import *

# define constants
mu_water = 0.001
mu_gas = 0.00001
rho_water = 1000
rho_gas = 0.001429

# Simulation parameters
re = 600
inflow_velocity = 0.05
gas_generate_rate = 0
outflow_pressure = 0
outflow_velocity_neu = 0
num_timesteps = 500
plot_interval = 25
dt = 0.001
dx = 0.1
dy = 0.1

# define the domain
INTERIOR = 0
WALL = 1
DOWN_GHOST = 2
UP_GHOST = 3
LEFT_GHOST = 4
RIGHT_GHOST = 5
BOUNDARY = 6
INLET = 7
OUTLET = 8
domain_index = {'interior': INTERIOR, 'wall': WALL, 'down_ghost': DOWN_GHOST, 'up_ghost': UP_GHOST, 'left_ghost': LEFT_GHOST, 'right_ghost': RIGHT_GHOST, 'boundary': BOUNDARY, 'inlet': INLET, 'outlet': OUTLET}

# Load mesh
mesh_p = np.loadtxt("p_mesh_half.csv", delimiter=",", dtype = int)
mesh_u = np.loadtxt("u_mesh_half.csv", delimiter=",", dtype = int)
mesh_v = np.loadtxt("v_mesh_half.csv", delimiter=",", dtype = int)
    
mesh_p = mesh_p.transpose()
mesh_u = mesh_u.transpose()
mesh_v = mesh_v.transpose()

# Get mesh information
interior_u = np.where(mesh_u == INTERIOR) 
wall_u = np.where(mesh_u == WALL) 
boundary_u = np.where(mesh_u == BOUNDARY) 
down_gost_u = np.where(mesh_u == DOWN_GHOST) 
up_gost_u = np.where(mesh_u == UP_GHOST) 
inlet_u = np.where(mesh_u == INLET)
outlet_u = np.where(mesh_u == OUTLET)
    
interior_v = np.where(mesh_v == INTERIOR) 
wall_v = np.where(mesh_v == WALL) 
boundary_v = np.where(mesh_v == BOUNDARY)
left_gost_v = np.where(mesh_v == LEFT_GHOST) 
right_gost_v = np.where(mesh_v == RIGHT_GHOST) 
inlet_v = np.where(mesh_v == INLET)
outlet_v = np.where(mesh_v == OUTLET)

interior_p = np.where(mesh_p == INTERIOR) 
wall_p = np.where(mesh_p == WALL) 
boundary_p = np.where(mesh_p == BOUNDARY)
left_gost_p = np.where(mesh_p == LEFT_GHOST) 
right_gost_p = np.where(mesh_p == RIGHT_GHOST) 
down_gost_p = np.where(mesh_p == DOWN_GHOST) 
up_gost_p = np.where(mesh_p == UP_GHOST) 

p_boundary = {
    UP_GHOST: 0,
    DOWN_GHOST: 0,
    LEFT_GHOST: 0,
    RIGHT_GHOST: 0,
    OUTLET: 0,
    INLET: 0
}

p_boundary_type = {
    UP_GHOST: 'Neumann',
    DOWN_GHOST: 'Neumann',
    LEFT_GHOST: 'Neumann',
    RIGHT_GHOST: 'Neumann',
    OUTLET: 'Dirichlet',
    INLET: 'Direchlet'
}

# initialize the solution
u = np.zeros_like(mesh_u, dtype = float)
v = np.zeros_like(mesh_v, dtype = float)
p = np.zeros_like(mesh_p, dtype = float)

# initialize the boundary condition
u[wall_u] = 0
u[inlet_u] = 0
v[wall_v] = 0
v[inlet_v] = inflow_velocity  

# time loop
for t in range(num_timesteps):
    # update ghost cells
    u[down_gost_u] = -u[(down_gost_u[0], down_gost_u[1] + 1)]
    u[up_gost_u] = -u[(up_gost_u[0], up_gost_u[1] - 1)]
    v[left_gost_v] = -v[(left_gost_v[0] + 1, left_gost_v[1])]
    v[right_gost_v] = -v[(right_gost_v[0] - 1, right_gost_v[1])]

    # set boundary condition
    u[inlet_u] = 0
    v[inlet_v] = inflow_velocity  
    u[wall_u] = 0
    v[wall_v] = 0
    v[outlet_v] = v[outlet_v[0], outlet_v[1] + 1]
    u[outlet_u] = u[outlet_u[0], outlet_u[1] + 1]

    # Step 1: predict u_star and v_star
    u_star = u.copy()
    u_a = u.copy()
    u_b = u.copy()
    v_star = v.copy()
    v_a = v.copy()
    v_b = v.copy()

    # u_a: advection term
    u_a[interior_u] = u_a[interior_u] = -1.0 / dx / 4.0 * ((u[(interior_u[0] + 1, interior_u[1])] + u[interior_u]) ** 2 \
        - (u[interior_u] + u[(interior_u[0] - 1, interior_u[1])]) ** 2) \
        -1.0 / dx / 4.0 * ((u[(interior_u[0], interior_u[1] + 1)] + u[interior_u]) * (v[(interior_u[0] + 1, interior_u[1])] + v[interior_u]) \
        - (u[interior_u] + u[(interior_u[0], interior_u[1] - 1)]) * (v[(interior_u[0] + 1, interior_u[1] - 1)] + v[(interior_u[0], interior_u[1] - 1)]))
    
    # u_b: diffusion term
    u_b[interior_u] = mu_water / dx ** 2 * (u[(interior_u[0] + 1, interior_u[1])] - 2 * u[interior_u] + u[(interior_u[0] - 1, interior_u[1])]) \
        + mu_water / dy ** 2 * (u[(interior_u[0], interior_u[1] + 1)] - 2 * u[interior_u] + u[(interior_u[0], interior_u[1] - 1)])

    # v_a: advection term
    v_a[interior_v] = -1.0 / dy / 4.0 * ((v[(interior_v[0], interior_v[1] + 1)] + v[interior_v]) ** 2 \
        - (v[interior_v] + v[(interior_v[0], interior_v[1] - 1)]) ** 2) \
        -1.0 / dy / 4.0 * ((v[(interior_v[0] + 1, interior_v[1])] + v[interior_v]) * (u[(interior_v[0], interior_v[1] + 1)] + u[interior_v]) \
        - (v[interior_v] + v[(interior_v[0] - 1, interior_v[1])]) * (u[(interior_v[0] - 1, interior_v[1] + 1)] + u[(interior_v[0] - 1, interior_v[1])]))

    # v_b: diffusion term
    v_b[interior_v] = mu_water / dx ** 2 * (v[(interior_v[0] + 1, interior_v[1])] - 2 * v[interior_v] + v[(interior_v[0] - 1, interior_v[1])]) \
        + mu_water / dy ** 2 * (v[(interior_v[0], interior_v[1] + 1)] - 2 * v[interior_v] + v[(interior_v[0], interior_v[1] - 1)])
    
    # update u_star and v_star
    u_star[interior_u] = u[interior_u] + dt * (u_a[interior_u] + u_b[interior_u])
    v_star[interior_v] = v[interior_v] + dt * (v_a[interior_v] + v_b[interior_v])

    u_star[down_gost_u] = -u_star[(down_gost_u[0], down_gost_u[1] + 1)]
    u_star[up_gost_u] = -u_star[(up_gost_u[0], up_gost_u[1] - 1)]
    v_star[left_gost_v] = -v_star[(left_gost_v[0] + 1, left_gost_v[1])]
    v_star[right_gost_v] = -v_star[(right_gost_v[0] - 1, right_gost_v[1])]

    # Step 2: solve for p
    # construct the right hand side of the pressure equation
    rhs_p = np.zeros_like(mesh_p, dtype = float)
    rhs_p[interior_p] = rho_water / dt * ((u_star[interior_p] - u_star[(interior_p[0] - 1, interior_p[1])]) / dx \
        + (v_star[(interior_p[0], interior_p[1])] - v_star[(interior_p[0], interior_p[1] - 1)]) / dy)
    
    # call iterative solver
    print('iteration number: ', t)
    p = poisson_iterative_solver(mesh_p, domain_index, p_boundary, p_boundary_type, dx, dy, rhs_p, tol=1e-6, maxiter=1000)

    # Step 3: correct u_star and v_star
    u[interior_u] = u_star[interior_u] - dt / rho_water / dx * (p[(interior_u[0] + 1, interior_u[1])] - p[interior_u])
    v[interior_v] = v_star[interior_v] - dt / rho_water / dy * (p[(interior_v[0], interior_v[1] + 1)] - p[interior_v])

    if t % plot_interval == 0:
        u_comp = np.zeros_like((mesh_p.shape[0]-2, mesh_p.shape[1]-2), dtype = float)
        v_comp = np.zeros_like((mesh_p.shape[0]-2, mesh_p.shape[1]-2), dtype = float)
        u_comp = (u[0:-1, 1:-1] + u[1:, 1:-1]) / 2
        v_comp = (v[1:-1, 0:-1] + v[1:-1, 1:]) / 2

        velocity = np.sqrt(u_comp ** 2 + v_comp ** 2).transpose()

        plt.figure()
        p_line = p.transpose()[20:40, 41]
        x = np.linspace(0, 0.1, 20)
        plt.plot(x, p_line)
        plt.xlabel('x')
        plt.ylabel('p')
        plt.title('Pressure at y = 0.2')

        plt.figure()
        p_plot = p.transpose()
        X = np.linspace(0, dx, p_plot.shape[0])
        Y = np.linspace(0, dy, p_plot.shape[1])
        plt.contourf(Y, X, p_plot, 20)
        plt.title('Pressure at {}'.format(t))
        plt.colorbar()
        plt.figure()
        X1 = np.linspace(0, dx, velocity.shape[0])
        Y1 = np.linspace(0, dy, velocity.shape[1])
        plt.contourf(Y1, X1, velocity, 20)
        plt.title('Velocity at {}'.format(t))
        plt.colorbar()
        plt.show()

u_comp = np.zeros_like((mesh_p.shape[0]-2, mesh_p.shape[1]-2), dtype = float)
v_comp = np.zeros_like((mesh_p.shape[0]-2, mesh_p.shape[1]-2), dtype = float)
u_comp = (u[0:-1, 1:-1] + u[1:, 1:-1]) / 2
v_comp = (v[1:-1, 0:-1] + v[1:-1, 1:]) / 2

velocity = np.sqrt(u_comp ** 2 + v_comp ** 2).transpose()

p = p.transpose()
X = np.linspace(0, dx, p.shape[0])
Y = np.linspace(0, dy, p.shape[1])
plt.contourf(Y, X, p, 20)
plt.colorbar()
plt.figure()
X1 = np.linspace(0, dx, velocity.shape[0])
Y1 = np.linspace(0, dy, velocity.shape[1])
plt.contourf(Y1, X1, velocity, 20)
plt.colorbar()
plt.show()
    