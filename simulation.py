import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from functionUtils import *

n_a = 300
n_b = 300
m = 400

# Source
alpha = np.linspace(-np.pi/2, np.pi/2, n_a)
alphaP = np.linspace(-np.pi/2, np.pi/2, m)
sigmaI = np.deg2rad(1)

I = gaussian(alpha[:, None], alphaP, 1, sigmaI)

# Captor
beta = np.linspace(-np.pi/2, np.pi/2, n_b)
betaP = np.linspace(-np.pi/2, np.pi/2, m)
sigmaf = np.deg2rad(2)

f = gaussian(beta[:, None], betaP, 1, sigmaf)

deltaAlphaP = alphaP[1]-alphaP[0]
deltaBetaP = betaP[1]-betaP[0]

sigmaR = np.deg2rad(10)
R_true = gaussian(alphaP[:, None], betaP, 1, sigmaR)

MP_simulated = deltaBetaP * deltaAlphaP * (I @ R_true @ f.T)

MP_min = MP_simulated.min()
MP_simulated -= MP_min
MP_max = MP_simulated.max()
MP_simulated /= MP_max

# MP_simulated += np.random.uniform(-0.1, 0.1, MP_simulated.shape)

R, M, opt_params = optimize_parameters(I, f, MP_simulated, deltaAlphaP, initial_guess=[0.1, 0.0001], track_history=True)

# eta, epsilon = 0.2, 1e-9
# M = MatrixSolve(f, MP_simulated.T, deltaAlphaP).T
# 
# R = tikhonovSolve(I, M, eta, epsilon, deltaBetaP)

MP_pred = deltaBetaP * deltaAlphaP * (I @ R @ f.T)

plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['legend.fontsize'] = 'xx-large'
plt.rcParams['axes.labelsize'] = 'xx-large'
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rcParams['ytick.labelsize'] = 'xx-large'

plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(I, extent=np.rad2deg([alpha[0], alpha[-1], alphaP[-1], alphaP[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('$\\alpha (^\\circ)$')
plt.ylabel('$\\alpha\' (^\\circ)$')
plt.title('I')

plt.subplot(1, 2, 2)
plt.imshow(f, extent=np.rad2deg([beta[0], beta[-1], betaP[-1], betaP[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('$\\beta (^\\circ)$')
plt.ylabel('$\\beta\' (^\\circ)$')
plt.title('f')
plt.show()

plt.figure(figsize=(8, 6))

#plt.subplot(2, 2, 1)
plt.imshow(M, extent=np.rad2deg([alpha[0], alpha[-1], betaP[-1], betaP[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('$\\alpha (^\\circ)$')
plt.ylabel('$\\beta\' (^\\circ)$')
plt.title('Reflected Intensity M')
plt.show()

plt.figure(figsize=(8, 6))
#plt.subplot(2, 2, 2)
plt.imshow(R, extent=np.rad2deg([alphaP[0], alphaP[-1], betaP[-1], betaP[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel('$\\alpha\' (^\\circ)$')
plt.ylabel('$\\beta\' (^\\circ)$')
plt.title(f'Intensity of Light Emitted by Surface R$(\\alpha\', \\beta\')$ \n $\\eta: {opt_params[0]:.3e}$  $\\epsilon: {opt_params[1]:.3e}$')
plt.savefig("reconstructed-R.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
#plt.subplot(2, 2, 3)
plt.imshow(MP_simulated, extent=np.rad2deg([alpha[0], alpha[-1], beta[-1], beta[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel('$\\alpha (^\\circ)$')
plt.ylabel('$\\beta (^\\circ)$')
plt.title('Simulated Measured Intensity M\'$(\\alpha, \\beta)$')
plt.savefig("simulated-mp.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
#plt.subplot(2, 2, 4)
plt.imshow(MP_pred, extent=np.rad2deg([alpha[0], alpha[-1], beta[-1], beta[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel('$\\alpha (^\\circ)$')
plt.ylabel('$\\beta\' (^\\circ)$')
plt.title('Reconstructed Measured Intensity M\'$(\\alpha, \\beta)$')
plt.savefig("reconstructed-mp.png", dpi=300, bbox_inches="tight")

plt.show()
