import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from functionUtils import *

n_a = 70
n_b = 50
m = 100

# Source
alpha = np.linspace(-np.pi/2, np.pi/2, n_a)
alphaP = np.linspace(-np.pi/2, np.pi/2, m)
sigmaI = np.deg2rad(1)

I = gaussian(alpha[:, None], alphaP, 15, sigmaI)

# Captor
beta = np.linspace(-np.pi/2, np.pi/2, n_b)
betaP = np.linspace(-np.pi/2, np.pi/2, m)
sigmaf = np.deg2rad(2)

f = gaussian(beta[:, None] - 0.013, betaP, 1, sigmaf)

deltaAlphaP = alphaP[1]-alphaP[0]
deltaBetaP = betaP[1]-betaP[0]

sigmaR = np.deg2rad(10)
R_true = gaussian(alphaP[:, None], betaP, 1, sigmaR)

MP_simulated = deltaBetaP * deltaAlphaP * (I @ R_true @ f.T)

MP_min = MP_simulated.min()
MP_simulated -= MP_min
MP_max = MP_simulated.max()
MP_simulated /= MP_max

initial_eta, initial_epsilon = 0.004, 0.001
# Plot Error with varying Eta
plt.figure(figsize=(12, 5))
etaVals = np.linspace(1e-09, 2, 200)
errors = [calculate_error([eta, initial_epsilon], I, f, MP_simulated, deltaAlphaP) for eta in etaVals]
plt.plot(etaVals, errors, linestyle='-', label=f'Error (deg)')
plt.xlabel('Eta')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error MP_simulated vs MP_pred')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()


# Plot Error with varying Epsilon
plt.figure(figsize=(12, 5))
epsilonVals = np.linspace(1e-09, 10, 100)
errors = [calculate_error([initial_eta, epsilon], I, f, MP_simulated, deltaAlphaP) for epsilon in epsilonVals]
plt.plot(epsilonVals, errors, linestyle='-', label=f'Error (deg)')
plt.xlabel('Epsilon')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error MP_simulated vs MP_pred')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()
