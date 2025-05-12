import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functionUtils import gaussian, optimize_parameters, readData

params = ()

# REACREATE THE UPPER HALF OF DATA!

with open("calibration.txt", "r") as file:
    params = file.read().split(" ")

sigma_I, C_f, sigma_f, beta_offset = np.float32(params)

alpha, beta, rawIntensity, MP  = readData("Beetle6", smooth=True)

delta = 1 / np.sqrt(alpha.shape[0]*beta.shape[0])
print(delta)

alphaP = np.linspace(-np.pi/2, np.pi/2, round(1/delta))

I = gaussian(alpha[:, None], alphaP, 1, sigma_I)

# Captor
betaP = np.linspace(-np.pi/2, np.pi/2, round(1/delta))

f = gaussian(beta[:, None] + beta_offset, betaP, C_f, sigma_f)

MP_min = MP.min()
MP -= MP_min
MP_max = MP.max()
MP /= MP_max

R, M, opt_params = optimize_parameters(I, f, MP, delta, [0.01, 1e-6], track_history=True)

MP_simulated = (delta**2) * (I @ R @ f.T) * MP_max + MP_min

plt.figure(figsize=(12, 5))

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

plt.figure(figsize=(12, 5))

plt.subplot(2, 2, 1)
plt.imshow(M, extent=np.rad2deg([alpha[0], alpha[-1], betaP[-1], betaP[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('$\\alpha (^\\circ)$')
plt.ylabel('$\\beta\' (^\\circ)$')
plt.title('Reflected Intensity M')

plt.subplot(2, 2, 2)
plt.imshow(R, extent=np.rad2deg([alphaP[0], alphaP[-1], betaP[-1], betaP[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('$\\alpha\' (^\\circ)$')
plt.ylabel('$\\beta\' (^\\circ)$')
plt.title('R')

plt.subplot(2, 2, 3)
plt.imshow(MP * MP_max + MP_min, extent=np.rad2deg([alpha[0], alpha[-1], beta[-1], beta[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('$\\alpha (^\\circ)$')
plt.ylabel('$\\beta (^\\circ)$')
plt.title('Smoothed data M\'')

plt.subplot(2, 2, 4)
plt.imshow(MP_simulated, extent=np.rad2deg([alpha[0], alpha[-1], beta[-1], beta[0]]), aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('$\\alpha (^\\circ)$')
plt.ylabel('$\\beta (^\\circ)$')
plt.title('Reconstructed Data M\'')

plt.show()

#  # Create a meshgrid for 3D plotting
# Alpha, Beta  = np.rad2deg(np.meshgrid(alpha, beta, indexing='ij'))
# 
# # Plot
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_proj_type("ortho")
# 
# # Surface plot
# surf = ax.plot_surface(Alpha, Beta, MP, cmap='plasma', edgecolor='none')
# 
# # Labels and title
# ax.set_xlabel('$\\beta \\,(^\\circ)$', labelpad=10)
# ax.set_ylabel('$\\alpha \\,(^\\circ)$', labelpad=10)
# ax.set_zlabel('M\'$(\\beta, \\alpha)$')
# ax.set_title("Measured Intensity")
# 
# # Colorbar
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Amplitude')
# 
# # Adjust view angle
# ax.view_init(elev=45, azim=30)  # Elevation and azimuthal angles
# 
# plt.tight_layout()
# plt.show()

# 
# # Plot
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_proj_type("ortho")
# 
# # Surface plot
# surf = ax.plot_surface(Beta, Alpha, MP_simulated, cmap='viridis', edgecolor='none')
# 
# # Labels and title
# ax.set_xlabel('$\\beta \\,(^\\circ)$', labelpad=10)
# ax.set_ylabel('$\\alpha \\,(^\\circ)$', labelpad=10)
# ax.set_zlabel('M\'$(\\beta, \\alpha)$')
# ax.set_title("Measured Intensity")
# 
# # Colorbar
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Amplitude')
# 
# # Adjust view angle
# ax.view_init(elev=45, azim=30)  # Elevation and azimuthal angles
# 
# plt.tight_layout()
# plt.show()
