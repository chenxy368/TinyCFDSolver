import numpy as np

def poisson_iterative_solver(domain_matrix, domain_index, boundary_condition, boundary_condition_type, dx, dy, f, tol, maxiter):
    """
    Solve the domain matrix using the iterative method
    Input:
        domain_matrix: the matrix denoting the domain
        boundary_condition: a dictionary containing the boundary condition values and respective indexes
        boundary_condition_type: a dictionary containing the boundary condition types and respective indexes (Dirichlet or Neumann)
        dx: the grid size
        dy: the grid size
        f: the right hand side of the equation
        tol: tolerance
        maxiter: maximum number of iterations
    Output:
        X: the solution matrix
    """
    # initialize the solution matrix
    INTERIOR = domain_index['interior']
    DOWN_GHOST = domain_index['down_ghost']
    UP_GHOST = domain_index['up_ghost']
    LEFT_GHOST = domain_index['left_ghost']
    RIGHT_GHOST = domain_index['right_ghost']
    INLET = domain_index['inlet']
    OUTLET = domain_index['outlet']
    WALL = domain_index['wall']
    
    X = np.zeros_like(domain_matrix)
    interior = np.where(domain_matrix == INTERIOR)
    down_ghost = np.where(domain_matrix == DOWN_GHOST)
    up_ghost = np.where(domain_matrix == UP_GHOST)
    left_ghost = np.where(domain_matrix == LEFT_GHOST)
    right_ghost = np.where(domain_matrix == RIGHT_GHOST)
    inlet = np.where(domain_matrix == INLET)
    outlet = np.where(domain_matrix == OUTLET)

    # initialize the errors
    error = 1
    iter = 0

    # initialize the boundary condition
    for key, value in boundary_condition_type.items():
        if value == 'Dirichlet':
                X[domain_matrix == key] = boundary_condition[key]
        elif value == 'Neumann':
            if key == UP_GHOST:
                X[domain_matrix == key] = X[(up_ghost[0], up_ghost[1] - 1)] - boundary_condition[key] * dy
            elif key == DOWN_GHOST:
                X[domain_matrix == key] = X[(down_ghost[0], down_ghost[1] + 1)] + boundary_condition[key] * dy
            elif key == LEFT_GHOST:
                X[domain_matrix == key] = X[(left_ghost[0] + 1, left_ghost[1])] - boundary_condition[key] * dx
            elif key == RIGHT_GHOST:
                X[domain_matrix == key] = X[(right_ghost[0] - 1, right_ghost[1])] + boundary_condition[key] * dx
            elif key == INLET:
                X[domain_matrix == key] = X[(inlet[0], inlet[1] - 1)] - boundary_condition[key] * dy
            elif key == OUTLET:
                X[domain_matrix == key] = X[(outlet[0], outlet[1] + 1)] + boundary_condition[key] * dy

    # iterate until the error is less than the tolerance or the maximum number of iterations is reached
    # while error > tol and iter < maxiter:
    while error > tol:
        X_ref = X.copy()

        # update the interior points
        X[interior] = 1 / 4 * (X[interior[0] + 1, interior[1]] + X[interior[0] - 1, interior[1]] + X[interior[0], interior[1] + 1] + X[interior[0], interior[1] - 1]) - \
            dx ** 2 / 4 * f[interior]

        # apply boundary conditions
        for key, value in boundary_condition_type.items():
            if value == 'Dirichlet':
                X[domain_matrix == key] = boundary_condition[key]
            elif value == 'Neumann':
                if key == UP_GHOST:
                    X[domain_matrix == key] = X[(up_ghost[0], up_ghost[1] - 1)] - boundary_condition[key] * dy
                elif key == DOWN_GHOST:
                    X[domain_matrix == key] = X[(down_ghost[0], down_ghost[1] + 1)] + boundary_condition[key] * dy
                elif key == LEFT_GHOST:
                    X[domain_matrix == key] = X[(left_ghost[0] + 1, left_ghost[1])] - boundary_condition[key] * dx
                elif key == RIGHT_GHOST:
                    X[domain_matrix == key] = X[(right_ghost[0] - 1, right_ghost[1])] + boundary_condition[key] * dx
                elif key == INLET:
                    X[domain_matrix == key] = X[(inlet[0], inlet[1] - 1)] - boundary_condition[key] * dy
                elif key == OUTLET:
                    X[domain_matrix == key] = X[(outlet[0], outlet[1] + 1)] + boundary_condition[key] * dy

        # calculate the error
        error = np.max(np.abs(X - X_ref))

        # update the iteration
        iter += 1
    print('Number of iterations: ', iter)

    return X