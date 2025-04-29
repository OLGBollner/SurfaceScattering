import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tikhonov import calculate_error, tikhonovSolve, optimize_parameters

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

betaPIndex = np.round(np.linspace(0, n_samples-1, 5)).astype("int")
R = np.zeros((R_true.shape[0], betaPIndex.shape[0]))

initial_eta, initial_epsilon = 0.004, 0.001
for index, betaPVal in enumerate(betaPIndex):
    optimized_eta, optimized_epsilon = optimize_parameters(I, f, MP_simulated, deltaAlphaP, deltaBetaP, betaPVal, R_true, [initial_eta, initial_epsilon])

    #Optimal eta: 0.025792322391461498, Optimal epsilon: 1e-09
    #optimized_eta = 0.025792322391461498
    #optimized_epsilon = 1e-09

    R[:, index], M = tikhonovSolve(I, f, MP_simulated, optimized_eta, optimized_epsilon, deltaAlphaP, deltaBetaP, betaPVal)

# Plot I and f
plt.figure(figsize=(12, 5))

plt.subplot(2, 2, 1)
plt.imshow(I, extent=np.rad2deg([alpha[0], alpha[-1], alphaP[-1], alphaP[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('Alpha (deg)')
plt.ylabel('AlphaP (deg)')
plt.title('Source Intensity I')

plt.subplot(2, 2, 2)
plt.imshow(f, extent=np.rad2deg([beta[0], beta[-1], betaP[-1], betaP[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('Beta (deg)')
plt.ylabel('BetaP (deg)')
plt.title('Captor Response f')

# Plot MP_simulated
plt.subplot(2, 2, 3)
plt.imshow(MP_simulated, extent=np.rad2deg([alpha[0], alpha[-1], beta[0], beta[-1]]), aspect='auto', cmap='viridis')
plt.colorbar(label='MP_simulated (Measured Intensity)')
plt.xlabel('Alpha (deg)')
plt.ylabel('Beta (deg)')
plt.title('Measured Intensity MP_simulated')

plt.tight_layout()
plt.show()

# Plot R
plt.figure(figsize=(12, 5))
for index, betaPVal in enumerate(betaPIndex):
    color = plt.cm.tab10(index % betaPIndex.shape[0])
    plt.plot(np.rad2deg(alphaP), R[:, index], color=color, linestyle='-', label=f'Reflectivity R, BetaP {np.rad2deg(betaP[betaPVal]):.1f} (deg)')
    plt.plot(np.rad2deg(alphaP), R_true[:, betaPVal], color=color, linestyle='-.', label=f'Generated R_true, BetaP {np.rad2deg(betaP[betaPVal]):.1f} (deg)')
plt.xlabel('AlphaP (deg)')
plt.ylabel('Normalized Reflectivity')
plt.title('Reflectivity R')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()
