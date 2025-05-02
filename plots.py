import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tikhonov import *
from initDistribs import *

betaVal = 0
alphaVal = 0
# Plot Error with varying Eta
plt.figure(figsize=(12, 5))
etaVals = np.linspace(1e-09, 2, 300)
errors = [calculate_error([eta, initial_epsilon], I, f, MP_simulated, deltaAlphaP, deltaBetaP, alphaVal, betaVal, R_true) for eta in etaVals]
plt.plot(etaVals, errors, linestyle='-', label=f'Error, BetaP {np.rad2deg(beta[betaVal]):.1f} (deg)')
plt.xlabel('Eta')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error R vs R_true')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()


# Plot Error with varying Epsilon
plt.figure(figsize=(12, 5))
epsilonVals = np.linspace(1e-06, 10, 300)
errors = [calculate_error([initial_eta, epsilon], I, f, MP_simulated, deltaAlphaP, deltaBetaP, alphaVal, betaVal, R_true) for epsilon in epsilonVals]
plt.plot(epsilonVals, errors, linestyle='-', label=f'Error, BetaP {np.rad2deg(beta[betaVal]):.1f} (deg)')
plt.xlabel('Epsilon')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error R vs R_true')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()

