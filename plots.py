import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tikhonov import *
from initDistribs import *

"""
# Plot Error with varying Eta
plt.figure(figsize=(12, 5))
etaVals = np.linspace(1e-09, 2, 100)
errors = [calculate_error([eta, initial_epsilon], I, f, MP_simulated, deltaAlphaP, deltaBetaP, R_true) for eta in etaVals]
plt.plot(etaVals, errors, linestyle='-', label=f'Error')
plt.xlabel('Eta')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error R vs R_true')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()


# Plot Error with varying Epsilon
plt.figure(figsize=(12, 5))
epsilonVals = np.linspace(1e-06, 10, 100)
errors = [calculate_error([initial_eta, epsilon], I, f, MP_simulated, deltaAlphaP, deltaBetaP, R_true) for epsilon in epsilonVals]
plt.plot(epsilonVals, errors, linestyle='-', label=f'Error')
plt.xlabel('Epsilon')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error R vs R_true')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()
"""

R_sol = tikhonovSolve(initial_eta, initial_epsilon, I, f, MP_simulated, deltaAlphaP, deltaBetaP)

print("Row sums:", R_sol.sum(axis=1))
print(f"Row sums of R (should be {1/ (deltaAlphaP * deltaBetaP):.2f}):")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

im0 = axs[0].imshow(R_true, aspect='auto', origin='lower')
axs[0].set_title("True R")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(R_sol, aspect='auto', origin='lower')
axs[1].set_title("Recovered R")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(np.abs(R_sol - R_true), aspect='auto', origin='lower')
axs[2].set_title("Abs Error in R")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()
