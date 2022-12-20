import numpy as np

def poisson_iterative_solver(domain, shape, boundaries, dx, dy, f, tol):
    X = np.zeros([shape[0], shape[1]], dtype = float)
    # initialize the errors
    error = 1
    iteration = 0

    # iterate until the error is less than the tolerance or the maximum number of iterations is reached
    # while error > tol and iter < maxiter:
    while error > tol:
        for boundary in boundaries:
            X = boundary.process(X)
        
        X_ref = X.copy()
        
        # update the interior points
        X[domain] = 1 / 4 * (X[domain[0] + 1, domain[1]] + X[domain[0] - 1, domain[1]] + X[domain[0], domain[1] + 1] + X[domain[0], domain[1] - 1]) - \
            dx ** 2 / 4 * f[domain]

        # calculate the error
        error = np.max(np.abs(X[domain] - X_ref[domain]))
        
        # update the iteration
        iteration += 1
    print('Number of iterations: ', iteration)

    return X