import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from pathlib import Path

def gaussian(x0, x, C, sigma):
    f = C * np.exp(-(np.pow(x-x0, 2))/(2 * sigma**2))
    return f

# Global variables to store history
R_history = []
iteration_count = [0, 0]
plot_initialized = False
if not plot_initialized:
    plt.figure(figsize=(10, 5))
    plot_initialized = True

def tikhonovSolve(I, M, eta, epsilon, delta, track_history=False):
    m, n = I.shape[1], M.shape[1]

    R = np.zeros((m, n))
    for betaP in range(n):
    
        # Find R
        A = np.vstack([delta * I, eta * np.eye(m)])
        b = np.concatenate([M[:, betaP], np.zeros(m)])

        A_stacked = np.vstack([epsilon * A, np.ones(m)])
        b_stacked = np.concatenate([epsilon * b, 1 / delta * np.ones(1)])

        R[:, betaP], _ = nnls(A_stacked, b_stacked)
        
    if track_history:
        R_history.append(R.copy())
        iteration_count[0] += 1
        plot_current_R(R, iteration_count[0], eta, epsilon)

    return R

def plot_current_R(R, iteration, eta, epsilon):
    plt.clf()
    plt.imshow(R, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"R at iteration {iteration} $\\eta: {eta:.3e}$ $\\epsilon: {epsilon:.3e}$")
    plt.xlabel("$\\alpha'$")
    plt.ylabel("$\\beta'$")
    plt.show(block=False)
    plt.draw()  # Update the figure
    plt.pause(0.01)

def MatrixSolve(f, B, delta):
    f_plus = np.linalg.pinv(f)
    A = 1/delta * f_plus @ B

    return A

def calculate_error(params, I, f, MP, delta, track_history=False):
    """
    Helper function to calculate the error for given eta and epsilon
    """
    eta1, eta2, epsilon1, epsilon2 = params

    M = tikhonovSolve(f, MP.T, eta1, epsilon1, delta).T

    R = tikhonovSolve(I, M, eta2, epsilon2, delta)

    MP_pred = delta * (I @ R @ f.T)
    MP_pred /= MP_pred.max()

    error = np.mean(np.linalg.norm(MP - MP_pred)/np.linalg.norm(MP_pred))

    if track_history:
        iteration_count[1] += 1
        print(f"Error: {error:.3e}, Iteration: {iteration_count[1]}", end="\r")
    return error

def optimize_parameters(I, f, MP, delta, initial_guess=[0.01, 0.01, 0.01, 0.01], track_history=False):
    """
    Optimize eta and epsilon to minimize the error
    """
    # Define bounds for parameters (eta and epsilon should typically be positive)
    bounds = [(1e-6, 10), (1e-9, 1), (1e-6, 10), (1e-9, 1)]
    
    # Run optimization
    result = minimize(
        calculate_error,
        x0=initial_guess,
        args=(I, f, MP, delta, track_history),
        bounds=bounds,
        method='L-BFGS-B',
    )
    
    if result.success:
        eta1_opt, eta2_opt, epsilon1_opt, epsilon2_opt = result.x
        min_error = result.fun
        print(f"Optimization successful. Minimum error: {min_error}")
        print(f"Optimal eta_1: {eta1_opt}, Optimal epsilon_1: {epsilon1_opt}")
        print(f"Optimal eta_2: {eta2_opt}, Optimal epsilon_2: {epsilon2_opt}")

        M = tikhonovSolve(f, MP.T, eta1_opt, epsilon2_opt, delta).T
        R = tikhonovSolve(I, M, eta2_opt, epsilon2_opt, delta)

        return R, M, (eta1_opt, eta2_opt, epsilon1_opt, epsilon2_opt)
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")

def readData(directorypath, smooth=False):
    directory = Path(directorypath)

    points = []
    for file in directory.iterdir():
        alpha = float(file.name[0:file.name.index("_")])
        beta = float(file.name[file.name.index("_")+1:file.name.index(".txt")]) - alpha

        data = np.loadtxt(file, comments="#", usecols=1)
        meanIntensity = np.mean(data)

        points.append((alpha, beta, meanIntensity))

    points = np.array(points)

    # Get unique, sorted alpha and beta values
    alpha_values = np.unique(points[:, 0])
    beta_values = np.unique(points[:, 1])
    
    # Create 2D intensity matrix
    intensity_matrix = np.full((alpha_values.shape[0], beta_values.shape[0]), 0.0)

    alpha_indices = np.searchsorted(alpha_values, points[:, 0])
    beta_indices = np.searchsorted(beta_values, points[:, 1])

    # Fill the matrix using the precomputed indices
    intensity_matrix[alpha_indices, beta_indices] = points[:, 2]

    smoothed = intensity_matrix.copy()
    if smooth:
        # First, replace zeros with NaN
        smoothed[smoothed == 0] = np.nan

        # Interpolate NaNs using nearest neighbors
        x, y = np.mgrid[:smoothed.shape[0], :smoothed.shape[1]]
        valid_mask = ~np.isnan(smoothed)
        smoothed = interpolate.griddata(
            (x[valid_mask], y[valid_mask]), 
            smoothed[valid_mask], 
            (x, y), 
            method='cubic'
        )

        # Optional: Apply Gaussian smoothing
        smoothed = gaussian_filter(smoothed, sigma=1)

        smoothed = np.nan_to_num(smoothed, nan=points[:, 2].min())

        #smoothed = gaussian_filter(smoothed, sigma=5)

    return np.deg2rad(alpha_values), np.deg2rad(beta_values), intensity_matrix, smoothed
