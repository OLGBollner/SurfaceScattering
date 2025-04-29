import numpy as np
from scipy.optimize import nnls, minimize

def tikhonovSolve(I, f, MP, eta, epsilon, deltaAlphaP, deltaBetaP, beta):
    # Find M
    B = np.vstack([deltaBetaP * f, eta * np.eye(f.shape[0])])
    b = np.concatenate([MP[:, beta], np.zeros(MP[:, beta].shape[0])])

    B_stacked = np.vstack([epsilon * B, np.ones(B.shape[1])])
    b_stacked = np.concatenate([epsilon * b, 1 / deltaBetaP * np.ones(1)])

    M, _ = nnls(B_stacked, b_stacked)
    M /= M.max()

    # Find R
    A = np.vstack([deltaAlphaP * I, eta * np.eye(I.shape[0])])
    a = np.concatenate([M, np.zeros(M.shape[0])])

    A_stacked = np.vstack([epsilon * A, np.ones(A.shape[1])])
    a_stacked = np.concatenate([epsilon * a, 1 / deltaAlphaP * np.ones(1)])

    R, _ = nnls(A_stacked, a_stacked)
    R /= R.max()
    
    return R, M

def calculate_error(params, I, f, MP_simulated, deltaAlphaP, deltaBetaP, betaVal, R_truth):
    """
    Helper function to calculate the error for given eta and epsilon
    """
    eta, epsilon = params
    R, M = tikhonovSolve(I, f, MP_simulated, eta, epsilon, deltaAlphaP, deltaBetaP, betaVal)
    error = np.abs(R - R_truth[:, betaVal])**2
    return np.mean(error)

def optimize_parameters(I, f, MP, deltaAlphaP, deltaBetaP, betaVal, R_truth, initial_guess=[0.01, 0.01]):
    """
    Optimize eta and epsilon to minimize the error
    """
    # Define bounds for parameters (eta and epsilon should typically be positive)
    bounds = [(1e-6, None), (1e-9, None)]
    
    # Run optimization
    result = minimize(
        calculate_error,
        x0=initial_guess,
        args=(I, f, MP, deltaAlphaP, deltaBetaP, betaVal, R_truth),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    if result.success:
        optimized_eta, optimized_epsilon = result.x
        min_error = result.fun
        print(f"Optimization successful. Minimum error: {min_error}")
        print(f"Optimal eta: {optimized_eta}, Optimal epsilon: {optimized_epsilon}")

        return optimized_eta, optimized_epsilon
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")
