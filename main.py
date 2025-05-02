import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tikhonov import calculate_error, tikhonovSolve, optimize_parameters
from initDistribs import *

for index, betaVal in enumerate(betaIndex):
    for index, alphaVal in enumerate(alphaIndex):
        optimized_eta, optimized_epsilon = optimize_parameters(I, f, MP_simulated, deltaAlphaP, deltaBetaP, alphaVal, betaVal, R_true, [initial_eta, initial_epsilon])

        R[:, index], M = tikhonovSolve(I, f, MP_simulated, initial_eta, initial_epsilon, deltaAlphaP, deltaBetaP, alphaVal, betaVal)

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
for index, betaVal in enumerate(betaIndex):
    color = plt.cm.tab10(index % betaIndex.shape[0])
    plt.plot(np.rad2deg(alphaP), R[:, index], color=color, linestyle='-', label=f'Reflectivity R, BetaP {np.rad2deg(beta[betaVal]):.1f} (deg)')
    plt.plot(np.rad2deg(alphaP), R_true[:, betaVal], color=color, linestyle='-.', label=f'Generated R_true, BetaP {np.rad2deg(beta[betaVal]):.1f} (deg)')
plt.xlabel('AlphaP (deg)')
plt.ylabel('Normalized Reflectivity')
plt.title('Reflectivity R')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()
