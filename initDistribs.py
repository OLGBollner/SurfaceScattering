import numpy as np
from scipy.stats import norm

n_angles = 20
n_primes = 40

# Source
alpha = np.linspace(np.pi/6, np.pi/2, n_angles)
alphaP = np.linspace(-np.pi/4, np.pi/4, n_primes)
sigmaI = np.deg2rad(5)

alpha_grid, alphaP_grid = np.meshgrid(alphaP, alpha)
I = norm.pdf(alpha_grid - alphaP_grid, sigmaI)

# Captor
beta = np.linspace(np.pi/6, np.pi/2, n_angles)
betaP = np.linspace(-np.pi/4, np.pi/4, n_primes)
sigmaf = np.deg2rad(10)

beta_grid, betaP_grid = np.meshgrid(betaP, beta)
f = norm.pdf(beta_grid - betaP_grid, sigmaf)

deltaAlphaP = alphaP[1]-alphaP[0]
deltaBetaP = betaP[1]-betaP[0]

# Ground truth R
alphaP_grid, betaP_grid = np.meshgrid(alphaP, betaP)
R_true = norm.pdf(alphaP_grid - betaP_grid, np.deg2rad(10))
R_true /= (np.sum(R_true) * (deltaAlphaP * deltaBetaP))

# Simulate MP using R_truth
MP_simulated = deltaBetaP * deltaAlphaP * I @ R_true @ f.T

alphaIndex = np.round(np.linspace(n_angles/2, n_angles-1, 1)).astype("int")
betaIndex = np.round(np.linspace(n_angles/2, n_angles-1, 1)).astype("int")

initial_eta, initial_epsilon = 0.01, 0.001
