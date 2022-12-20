import numpy as np

from solvers import poisson_iterative_solver
from plot import plot_one_contourf, plot_one_streamlines, animate
from boundary import boundary_condition

# define the domain
INTERIOR = -1
WALL = 0
UP_GHOST = 1
DOWN_GHOST = 2
LEFT_GHOST = 3
RIGHT_GHOST = 4
BOUNDARY = 5
INLET = 6
OUTLET = 7

# define constants
mu_water = 0.001
rho_water = 1000

# Simulation parameters
inflow_velocity = 0.05
num_timesteps = 100
plot_interval = 3
dt = 0.001
dx = 0.1
dy = 0.1

# Load mesh
          
mesh_p = np.loadtxt("small_case/p_mesh.csv", delimiter=",", dtype = int)
mesh_u = np.loadtxt("small_case/u_mesh.csv", delimiter=",", dtype = int)
mesh_v = np.loadtxt("small_case/v_mesh.csv", delimiter=",", dtype = int)

mesh_p = np.flipud(mesh_p)
mesh_u = np.flipud(mesh_u)
mesh_v = np.flipud(mesh_v)
    
mesh_p = mesh_p.transpose()
mesh_u = mesh_u.transpose()
mesh_v = mesh_v.transpose()

# Get mesh information
interior_u = np.where(mesh_u == INTERIOR) 
interior_v = np.where(mesh_v == INTERIOR) 
interior_p = np.where(mesh_p == INTERIOR) 

interior_else_u = np.where(mesh_u != INTERIOR) 
interior_else_v = np.where(mesh_v != INTERIOR) 
interior_else_p = np.where(mesh_p != INTERIOR) 



u_boundaries = []
def u_wall_opt(u, domain):
    u[domain] = 0
    return u
u_boundaries.append(boundary_condition(WALL, "wall", np.where(mesh_u == WALL), u_wall_opt))
def u_boundary_opt(u, domain):
    u[domain] = 0
    return u
u_boundaries.append(boundary_condition(BOUNDARY, "boundary", np.where(mesh_u == BOUNDARY), u_boundary_opt))
def u_down_gost_opt(u, domain):
    u[domain] = -u[(domain[0], domain[1] + 1)]
    return u
u_boundaries.append(boundary_condition(DOWN_GHOST, "down ghost", np.where(mesh_u == DOWN_GHOST), u_down_gost_opt))
def u_up_gost_opt(u, domain):
    u[domain] = -u[(domain[0], domain[1] - 1)]
    return u
u_boundaries.append(boundary_condition(UP_GHOST, "up ghost", np.where(mesh_u == UP_GHOST), u_up_gost_opt))
def u_inlet_opt(u, domain):
    u[domain] = 0
    return u
u_boundaries.append(boundary_condition(INLET, "inlet", np.where(mesh_u == INLET), u_inlet_opt))
def u_outlet_opt(u, domain):
    u[domain] = u[domain[0], domain[1]-1]
    return u
u_boundaries.append(boundary_condition(OUTLET, "outlet", np.where(mesh_u == OUTLET), u_outlet_opt))


v_boundaries = []
def v_wall_opt(v, domain):
    v[domain] = 0
    return v
v_boundaries.append(boundary_condition(WALL, "wall", np.where(mesh_v == WALL), v_wall_opt))
def v_boundary_opt(v, domain):
    v[domain] = 0
    return v
v_boundaries.append(boundary_condition(BOUNDARY, "boundary", np.where(mesh_v == BOUNDARY), v_boundary_opt))
def v_left_gost_opt(v, domain):
    v[domain] = -v[(domain[0] + 1, domain[1])]
    return v
v_boundaries.append(boundary_condition(DOWN_GHOST, "down ghost", np.where(mesh_v == DOWN_GHOST), v_left_gost_opt))
def v_right_gost_opt(v, domain):
    v[domain] = -v[(domain[0] - 1, domain[1])]
    return v
v_boundaries.append(boundary_condition(UP_GHOST, "up ghost", np.where(mesh_v == UP_GHOST), v_right_gost_opt))
def v_inlet_opt(v, domain):
    v[domain] = inflow_velocity
    return v
v_boundaries.append(boundary_condition(INLET, "inlet", np.where(mesh_v == INLET), v_inlet_opt))
def v_outlet_opt(v, domain):
    v[domain] = v[domain[0], domain[1]-1]
    return v
v_boundaries.append(boundary_condition(OUTLET, "outlet", np.where(mesh_v == OUTLET), v_outlet_opt))    


p_boundaries = []

def p_wall_opt(p, domain):
    p[domain] = 0
    return p
p_boundaries.append(boundary_condition(WALL, "wall", np.where(mesh_p == WALL), p_wall_opt))
def p_boundary_opt(p, domain):
    p[domain] = 0
    return p
p_boundaries.append(boundary_condition(BOUNDARY, "boundary", np.where(mesh_p == BOUNDARY), p_boundary_opt))
def p_left_gost_opt(p, domain):
    p[domain] = p[(domain[0] + 1, domain[1])]
    return p
p_boundaries.append(boundary_condition(LEFT_GHOST, "left ghost", np.where(mesh_p == LEFT_GHOST), p_left_gost_opt))
def p_right_gost_opt(p, domain):
    p[domain] = p[(domain[0] - 1, domain[1])]
    return p
p_boundaries.append(boundary_condition(RIGHT_GHOST, "right ghost", np.where(mesh_p == RIGHT_GHOST), p_right_gost_opt))
def p_down_gost_opt(p, domain):
    p[domain] = p[(domain[0], domain[1] + 1)]
    return p
p_boundaries.append(boundary_condition(DOWN_GHOST, "down ghost", np.where(mesh_p == DOWN_GHOST), p_down_gost_opt))
def p_up_gost_opt(p, domain):
    p[domain] = p[(domain[0], domain[1] - 1)]
    return p
p_boundaries.append(boundary_condition(UP_GHOST, "up ghost", np.where(mesh_p == UP_GHOST), p_up_gost_opt))
def p_inlet_opt(p, domain):
    p[domain] = 0
    return p
p_boundaries.append(boundary_condition(INLET, "inlet", np.where(mesh_p == INLET), p_inlet_opt))
def p_outlet_opt(p, domain):
    p[domain] = 0
    return p
p_boundaries.append(boundary_condition(OUTLET, "outlet", np.where(mesh_p == OUTLET), p_outlet_opt))  

# initialize the solution
u = np.zeros_like(mesh_u, dtype = float)
v = np.zeros_like(mesh_v, dtype = float)
p = np.zeros_like(mesh_p, dtype = float)


# animation
velocity_list = []
pressure_list = []

# time loop
for t in range(num_timesteps):
    for boundary in u_boundaries:
        u = boundary.process(u)
    for boundary in v_boundaries:
        v = boundary.process(v)
    
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


    for boundary in u_boundaries:
        u_star = boundary.process(u_star)
    for boundary in v_boundaries:
        v_star = boundary.process(v_star)
    # Step 2: solve for p
    # construct the right hand side of the pressure equation
    rhs_p = np.zeros_like(mesh_p, dtype = float)
    rhs_p[interior_p] = rho_water / dt * ((u_star[interior_p] - u_star[(interior_p[0] - 1, interior_p[1])]) / dx \
        + (v_star[(interior_p[0], interior_p[1])] - v_star[(interior_p[0], interior_p[1] - 1)]) / dy)
    
    # call iterative solver
    print('iteration number: ', t)
    p = poisson_iterative_solver(interior_p, p.shape, p_boundaries, dx, dy, rhs_p, tol=1e-1)

    # Step 3: correct u_star and v_star
    u[interior_u] = u_star[interior_u] - dt / rho_water / dx * (p[(interior_u[0] + 1, interior_u[1])] - p[interior_u])
    v[interior_v] = v_star[interior_v] - dt / rho_water / dy * (p[(interior_v[0], interior_v[1] + 1)] - p[interior_v])

    if (t + 1) % plot_interval == 0:
        u[interior_else_u] = 0
        v[interior_else_v] = 0
        p[interior_else_p] = 0
        
        u_comp = np.zeros_like((mesh_p.shape[0]-2, mesh_p.shape[1]-2), dtype = float)
        v_comp = np.zeros_like((mesh_p.shape[0]-2, mesh_p.shape[1]-2), dtype = float)
        u_comp = (u[0:-1, 1:-1] + u[1:, 1:-1]) / 2
        v_comp = (v[1:-1, 0:-1] + v[1:-1, 1:]) / 2


        velocity = np.sqrt(u_comp ** 2 + v_comp ** 2)
        velocity = velocity.transpose()
        velocity_list.append(velocity)
        pressure_list.append((np.transpose(p)))
        
        plot_one_contourf(velocity, dx, dy, "velocity magnitude at " + str(round((t + 1) * dt, 3)) + "s", "velocity[m/s]", 0.0, 0.07)        
        plot_one_contourf((np.transpose(p)), dx, dy, "pressure at " + str(round((t + 1) * dt, 3)) + "s", "pressure[Pa]", 0.0, 5000.0)        
        plot_one_streamlines(u_comp.transpose(), v_comp.transpose(), dx, dy, 'Streamlines at ' + str(round((t + 1) * dt, 3)) + 's')


animate(velocity_list, dx, dy, "velocity magnitude", "velocity[m/s]", 0.0, 0.07)
animate(pressure_list, dx, dy, "pressure", "pressure[Pa]", 0, 5000.0)

    