import numpy as np
from scipy.optimize import nnls, minimize
from scipy.sparse.linalg import LinearOperator, lsqr

def tikhonovSolve(eta, I, f, MP, deltaAlphaP, deltaBetaP):
    """
    Solve for R(α', β') and M(α, β') using NNLS with Tikhonov regularization and constraints.
    
    Parameters:
        I: Source function I(α, α') [n_α × n_α']
        f: Detector function f(β, β') [n_β × n_β']
        MP: Measured data M'(α, β) [n_α × n_β]
        eta: Tikhonov regularization parameter
        deltaAlphaP, deltaBetaP: Discretization steps (Δα', Δβ')
        beta: Fixed β angle index to solve for
    """

    n, m = I.shape

    # define matrix operations
    def A_matvec(x):
        """Compute A @ x without storing A."""
        # Reshape x into R (m × m)
        R = x.reshape((m, m))
        # Compute I @ R @ f.T
        temp = I @ R @ f.T
        # Scale by Δα' Δβ' and vectorize
        return (deltaAlphaP * deltaBetaP * temp).flatten()

    def A_rmatvec(y):
        """Compute A.T @ y without storing A."""
        # Reshape y into M' (n × n)
        M_prime = y.reshape((n, n))
        # Compute I.T @ M' @ f
        temp = I.T @ M_prime @ f
        # Scale by Δα' Δβ' and vectorize
        return (deltaAlphaP * deltaBetaP * temp).flatten()

    # Define A as a LinearOperator
    A = LinearOperator(
        shape=(n * n, m * m),  # A is n² × m²
        matvec=A_matvec,       # Computes A @ x
        rmatvec=A_rmatvec      # Computes A.T @ y
    )

    b = MP.flatten()

    x = lsqr(A, b, damp=np.sqrt(eta), iter_lim=1000)[0]
    R = x.reshape((m, m))
    R = np.maximum(R, 0)
    R /= (np.sum(R) * deltaAlphaP * deltaBetaP)
    
    return R

def calculate_error(eta, I, f, MP_simulated, deltaAlphaP, deltaBetaP, R_truth):
    """
    Helper function to calculate the error for given eta and epsilon
    """
    R = tikhonovSolve(eta, I, f, MP_simulated, deltaAlphaP, deltaBetaP)
    rmse = np.sqrt(np.mean((R - R_truth)**2))
    return rmse

def optimize_parameters(I, f, MP, deltaAlphaP, deltaBetaP, R_true, initial_guess=0.01):
    """
    Optimize eta and epsilon to minimize the error
    """
    # Define bounds for parameters (eta and epsilon should typically be positive)
    bounds = [(1e-6, None)]
    
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
