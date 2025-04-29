import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tikhonov import *

n_samples = 200

# Source
alpha = np.linspace(0, np.pi/2, n_samples)
alphaP = np.linspace(-np.pi/4, np.pi/4, n_samples)
sigmaI = np.deg2rad(5)

alpha_grid, alphaP_grid = np.meshgrid(alpha, alphaP)
I = norm.pdf(alpha_grid - alphaP_grid, sigmaI)
I /= I.max()

# Captor
beta = np.linspace(0, np.pi/2, n_samples)
betaP = np.linspace(-np.pi/4, np.pi/4, n_samples)
sigmaf = np.deg2rad(10)

beta_grid, betaP_grid = np.meshgrid(beta, betaP)
f = norm.pdf(beta_grid - betaP_grid, sigmaf)
f /= f.max()

# Measured Intensity emitted from alpha and reflected at beta prime
M = np.zeros((alpha.shape[0], betaP.shape[0]))

deltaAlphaP = alphaP[1]-alphaP[0]
deltaBetaP = betaP[1]-betaP[0]

# Ground truth R
alphaP_grid, betaP_grid = np.meshgrid(alphaP, betaP)
R_true = norm.pdf(alphaP_grid - betaP_grid, 0, np.deg2rad(10))
R_true /= R_true.max()

# Simulate MP using R_truth
MP_simulated = deltaBetaP * deltaAlphaP * I @ R_true @ f
#MP_simulated += 0.001 * np.random.randn(*MP_simulated.shape)
MP_simulated /= MP_simulated.max()

betaPIndex = np.round(np.linspace(0, n_samples-1, 4)).astype("int")
R = np.zeros((R_true.shape[0], betaPIndex.shape[0]))

initial_eta, initial_epsilon = 0.004, 0.001
betaPVal = 0
# Plot Error with varying Eta
plt.figure(figsize=(12, 5))
etaVals = np.linspace(1e-09, 1, 50)
errors = [calculate_error([eta, initial_epsilon], I, f, MP_simulated, deltaAlphaP, deltaBetaP, betaPVal, R_true) for eta in etaVals]
plt.plot(etaVals, errors, linestyle='-', label=f'Error, BetaP {np.rad2deg(betaP[betaPVal]):.1f} (deg)')
plt.xlabel('Eta')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error R vs R_true')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()


# Plot Error with varying Epsilon
plt.figure(figsize=(12, 5))
epsilonVals = np.linspace(1e-09, 10, 50)
errors = [calculate_error([initial_eta, epsilon], I, f, MP_simulated, deltaAlphaP, deltaBetaP, betaPVal, R_true) for epsilon in epsilonVals]
plt.plot(etaVals, errors, linestyle='-', label=f'Error, BetaP {np.rad2deg(betaP[betaPVal]):.1f} (deg)')
plt.xlabel('Epsilon')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error R vs R_true')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()
