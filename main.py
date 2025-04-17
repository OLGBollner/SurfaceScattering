import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Source
alpha = np.linspace(0, np.pi/2, 100)
alphaP = np.linspace(-np.pi/2, np.pi/2, 100)
sigmaI = np.deg2rad(5)

alpha_grid, alphaP_grid = np.meshgrid(alpha, alphaP)
I = norm.pdf(alpha_grid - alphaP_grid, sigmaI)
I /= I.max()

# Captor
beta = np.linspace(0, np.pi/2, 100)
betaP = np.linspace(-np.pi/2, np.pi/2, 100)
sigmaf = np.deg2rad(10)

beta_grid, betaP_grid = np.meshgrid(beta, betaP)
f = norm.pdf(beta_grid - betaP_grid, sigmaf)
f /= f.max()


