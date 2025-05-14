import numpy as np
import matplotlib.pyplot as plt
from functionUtils import *

n_a = 100
n_b = 100
m = 200

# Source
alpha = np.linspace(-np.pi/2, np.pi/2, n_a)
alphaP = np.linspace(-np.pi/2, np.pi/2, m)
sigmaI = np.deg2rad(5)

I = gaussian(alpha[:, None], alphaP, 1, sigmaI)

# Captor
beta = np.linspace(-np.pi/2, np.pi/2, n_b)
betaP = np.linspace(-np.pi/2, np.pi/2, m)
sigmaf = np.deg2rad(5)

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

initial_eta, initial_epsilon = 0.4, 1e-4


plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['legend.fontsize'] = 'xx-large'
plt.rcParams['axes.labelsize'] = 'xx-large'
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rcParams['ytick.labelsize'] = 'xx-large'

etaVals = np.linspace(1e-09, 2, 100)
errors = [calculate_error([eta, initial_epsilon], I, f, MP_simulated, deltaAlphaP, track_history=True) for eta in etaVals]

# Plot Error with varying Eta
plt.figure(figsize=(8, 6))
plt.plot(etaVals, errors, linestyle='-', label=f'Error $(^\\circ)$')
plt.xlabel('$\\eta$')
plt.ylabel('Relative error')
plt.title(f'Relative Error of M$\'_{{simulated}}$ vs M$\'_{{predicted}}$ \n $\\epsilon: {initial_epsilon:.3e}$, minimum error: {np.min(errors):.3f}')
plt.legend()
plt.grid()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("eta-error.png", dpi=300, bbox_inches="tight")
plt.show()


epsilonVals = np.linspace(1e-09, 10, 100)
errors = [calculate_error([initial_eta, epsilon], I, f, MP_simulated, deltaAlphaP, track_history=True) for epsilon in epsilonVals]

# Plot Error with varying Epsilon
plt.figure(figsize=(8, 6))
plt.plot(epsilonVals, errors, linestyle='-', label=f'Error $(^\\circ)$')
plt.xlabel('$\\epsilon$')
plt.ylabel('Relative error')
plt.title(f'Relative Error of M$\'_{{simulated}}$ vs M$\'_{{predicted}}$ \n $\\eta: {initial_eta:.3e}$, minimum error: {np.min(errors):.3f}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("epsilon-error.png", dpi=300, bbox_inches="tight")
plt.show()
