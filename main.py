import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tikhonov import calculate_error, tikhonovSolve, optimize_parameters

n_angles = 200
n_primeAngles = 400

# Angles
alpha = np.linspace(0, np.pi/2, n_angles)
alphaP = np.linspace(-np.pi/4, np.pi/4, n_primeAngles)

beta = np.linspace(0, np.pi/2, n_angles)
betaP = np.linspace(-np.pi/4, np.pi/4, n_primeAngles)

deltaAlphaP = alphaP[1]-alphaP[0]
deltaBetaP = betaP[1]-betaP[0]

# Source
sigmaI = np.deg2rad(5)
alpha_grid, alphaP_grid = np.meshgrid(alpha, alphaP, indexing="ij")
I = norm.pdf(alpha_grid - alphaP_grid, sigmaI)
I /= I.max()

# Captor
sigmaf = np.deg2rad(10)
beta_grid, betaP_grid = np.meshgrid(beta, betaP, indexing="ij")
f = norm.pdf(beta_grid - betaP_grid, sigmaf)
f /= f.max()

# Simulated R
alphaP_grid, betaP_grid = np.meshgrid(alphaP, betaP, indexing="ij")
R_true = norm.pdf(alphaP_grid - betaP_grid, 0, np.deg2rad(10))
R_true /= (np.sum(R_true) * deltaAlphaP * deltaBetaP)

# Simulate MP using R_true
MP_simulated = deltaBetaP * deltaAlphaP * I @ R_true @ f.T
MP_simulated /= MP_simulated.max()

print(f"MP: {MP_simulated.shape}")
print(f"R_true: {R_true.shape}")

#betaIndex = np.array([((n_samples-1)/5)]).astype("int")
#betaIndex = np.round(np.linspace(0, n_angles-1, 2)).astype("int")
#alphaIndex = np.round(np.linspace(0, n_angles-1, 2)).astype("int")

alphaIndex = np.round(n_angles/2).astype("int")
betaIndex = np.round(n_angles/2).astype("int")
alphaVal = alpha[alphaIndex]
betaVal = beta[betaIndex]

#R = np.zeros((R_true.shape[0], betaIndex.shape[0]))
#M = np.zeros((M_true.shape[0], betaIndex.shape[0]))

initial_eta = 0.0001
optimized_eta = optimize_parameters(I, f, MP_simulated, deltaAlphaP, deltaBetaP, R_true, initial_eta)

#Optimal eta: 0.025792322391461498, Optimal epsilon: 1e-09
#optimized_eta = 0.025792322391461498
#optimized_epsilon = 1e-09

R = tikhonovSolve(optimized_eta, I, f, MP_simulated, deltaAlphaP, deltaBetaP)
print(f"R: {R.shape}")

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

"""
# plot errors
eta_values = np.logspace(-6, 1, 50)
errors = [calculate_error(eta, I, f, MP_simulated, deltaAlphaP, deltaBetaP, R_true) for eta in eta_values]
plt.loglog(eta_values, errors)
plt.xlabel("eta")
plt.ylabel("RMSE")
plt.show()
"""

# Plot R
plt.figure(figsize=(12, 5))
plt.plot(np.rad2deg(alphaP), R[:, 0], linestyle='-', label=f'Reflectivity R, Beta {np.rad2deg(betaVal):.1f} (deg)')
plt.plot(np.rad2deg(alphaP), R_true[:, 0], linestyle='-.', label=f'Generated R_true, Beta {np.rad2deg(betaVal):.1f} (deg)')
plt.xlabel('AlphaP (deg)')
plt.ylabel('Normalized Reflectivity')
plt.title('Reflectivity R')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()

