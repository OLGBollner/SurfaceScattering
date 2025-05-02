import numpy as np
from scipy.optimize import nnls, minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lsqr, lsmr
from initDistribs import *

def tikhonovSolve(eta, epsilon, I, f, MP, deltaAlphaP, deltaBetaP):
    n, m = I.shape

    c_weight = 1

    b = MP.flatten()
    A = deltaAlphaP * deltaBetaP * np.kron(f, I)

    C = np.zeros((m, m * m))
    for i in range(m):
        C[i, i * m:(i + 1) * m] = 1
    c = np.full((m,), 1.0 / (deltaAlphaP * deltaBetaP))

    A_stacked = np.vstack([epsilon * A, epsilon * eta * np.eye(m**2), c_weight * C])
    b_stacked = np.concatenate([b, np.zeros(m**2), c_weight * c])

    x, _ = nnls(A_stacked, b_stacked)
    R = x.reshape((m, m))

    return R

def calculate_error(params, I, f, MP_simulated, deltaAlphaP, deltaBetaP, R_true):
    """
    Helper function to calculate the error for given eta and epsilon
    """
    eta, epsilon = params
    R = tikhonovSolve(eta, epsilon, I, f, MP_simulated, deltaAlphaP, deltaBetaP)
    rmse = np.sqrt(np.mean((R - R_true)**2))
    return rmse

def optimize_parameters(I, f, MP, deltaAlphaP, deltaBetaP, R_true, initial_guess=[0.01, 0.001]):
    """
    Optimize eta and epsilon to minimize the error
    """
    # Define bounds for parameters (eta and epsilon should typically be positive)
    bounds = [(1e-9, None), (1e-9, None)]
    
    # Run optimization
    result = minimize(
        calculate_error,
        x0=initial_guess,
        args=(I, f, MP, deltaAlphaP, deltaBetaP, R_true),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    if result.success:
        optimized_eta = result.x[0]
        min_error = result.fun
        print(f"Optimization successful. Minimum error: {min_error}")
        print(f"Optimal eta: {optimized_eta}")

        return optimized_eta
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")
