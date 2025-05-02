import numpy as np
from scipy.optimize import nnls, minimize

def tikhonovSolve(I, f, MP, eta, epsilon, deltaAlphaP, deltaBetaP, alpha, beta):
    n, m = I.shape

    # Find M
    B = np.vstack([deltaBetaP * f[beta, :], eta * np.eye(m)])
    b = np.concatenate([[MP[alpha, beta]], np.zeros((m, 1)).flatten()])

    B_stacked = np.vstack([epsilon * B, np.ones((1, m))])
    b_stacked = np.concatenate([epsilon * b, [1 / deltaBetaP]])

    M, _ = nnls(B_stacked, b_stacked)
    M /= M.max()

    # Find R
    A = np.vstack([deltaAlphaP * I[alpha, :], eta * np.eye(m)])
    a = np.concatenate([[M[alpha]], np.zeros((m, 1)).flatten()])

    A_stacked = np.vstack([epsilon * A, np.ones((1, m))])
    a_stacked = np.concatenate([epsilon * a, [1 / deltaAlphaP]])

    R, _ = nnls(A_stacked, a_stacked)
    R /= R.max()

    return R, M

def calculate_error(params, I, f, MP_simulated, deltaAlphaP, deltaBetaP, alphaVal, betaVal, R_truth):
    """
    Helper function to calculate the error for given eta and epsilon
    """
    eta, epsilon = params
    R, M = tikhonovSolve(I, f, MP_simulated, eta, epsilon, deltaAlphaP, deltaBetaP, alphaVal, betaVal)
    error = np.abs(R - R_truth[:, betaVal])**2
    return np.mean(error)

def optimize_parameters(I, f, MP, deltaAlphaP, deltaBetaP, alphaVal, betaVal, R_truth, initial_guess=[0.01, 0.01]):
    """
    Optimize eta and epsilon to minimize the error
    """
    # Define bounds for parameters (eta and epsilon should typically be positive)
    bounds = [(1e-6, None), (1e-9, None)]
    
    # Run optimization
    result = minimize(
        calculate_error,
        x0=initial_guess,
        args=(I, f, MP, deltaAlphaP, deltaBetaP, alphaVal, betaVal, R_truth),
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
