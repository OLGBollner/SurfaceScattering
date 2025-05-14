import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import least_squares
from functionUtils import gaussian, readData

def estimateSourceParams():
    diameter = np.array([0.075, 0.065, 0.052, 0.045, 0.035])
    length = np.array([0.965, 0.933, 0.800, 0.720, 0.625])

    maxDeviation = np.mean(np.arctan((diameter/2) / length))
    sigmaI = maxDeviation/3

    return maxDeviation, sigmaI

def generateSource(alpha, deltaAlphaP, C):
    _, sigmaI = estimateSourceParams()
    alpha_prime = np.linspace(-np.pi/18, np.pi/18, round(1/deltaAlphaP))

    I = gaussian(alpha, alpha_prime, C, sigmaI)

    return I, alpha_prime, sigmaI

def generateCaptor(beta, deltaAlphaP, sigmaf, C):
    alpha_prime = np.linspace(-np.pi/18, np.pi/18, round(1/deltaAlphaP))
    f = gaussian(beta[:, None], alpha_prime, C, sigmaf)

    return f

def residuals(params, I, beta, deltaAlphaP, MP):
    C, sigmaf, beta_offset = params

    f_init = generateCaptor(beta + beta_offset, deltaAlphaP, sigmaf, C)

    MP_pred = deltaAlphaP * (I @ f_init.T)

    return MP_pred - MP

def estimateCaptor(deltaAlphaP, I, MP, beta):
    
    initial_sigmaf, initial_C, initial_beta_offset = 0.04, 139, 0.02

    sol = least_squares(
        fun=residuals,
        x0=[initial_C, initial_sigmaf, initial_beta_offset],
        args=(I, beta, deltaAlphaP, MP),
        bounds=([1e-6, 1e-6, -np.pi], [np.inf, np.inf, np.pi])
    )
    C_opt, sigmaf_opt, beta_offset_opt = sol.x
    print(sol.message)

    f = generateCaptor(beta + beta_offset_opt, deltaAlphaP, sigmaf_opt, C_opt)

    return f, (C_opt, sigmaf_opt, beta_offset_opt)

plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['legend.fontsize'] = 'xx-large'
plt.rcParams['axes.labelsize'] = 'xx-large'
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rcParams['ytick.labelsize'] = 'xx-large'

alpha, beta, rawIntensity, MP = readData("Source Scattering alpha_prime2") # .1 deg step
deltaAlphaP = 0.001
alpha -= np.pi

MP_min = MP.min()
MP -= MP_min
MP_max = MP.max()
MP /= MP_max

MP = MP.flatten()

I, alpha_prime, sigmaI = generateSource(alpha, deltaAlphaP, 1)
f, opt_params = estimateCaptor(deltaAlphaP, I, MP, beta)

with open("calibration.txt", "w") as file:
    file.write(f"{sigmaI} {' '.join(map(str, opt_params))}")

M_pred = deltaAlphaP * (I @ f.T)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(beta * 180/np.pi, MP, linewidth=2, linestyle='-', color='r', label='M$_{measured}$ ')
ax.plot(beta * 180/np.pi, M_pred, linewidth=2, linestyle='-', color='g', label='M$_{reconstructed}$')
# Add labels and title
ax.set_xlabel('$\\beta \\, (^\\circ)$')
ax.set_ylabel('Intensity')
ax.set_title('Measured Average Intensity \n Over wavelengths 150 - 1200 nm and stepsize $\\Delta_{\\beta} = 0.1 ^\\circ$')
plt.legend(loc=(0.01, 0.7))
plt.grid(True)
xmin, xmax = -5, 5
ax.set_xticks(np.linspace(xmin, xmax, 5))
plt.savefig("measured-intensity-01step.png", dpi=300, bbox_inches="tight")

plt.show()


# Create a meshgrid for 3D plotting
Beta, Alpha = np.rad2deg(np.meshgrid(beta, alpha_prime, indexing='ij'))

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type("ortho")

# Surface plot
surf = ax.plot_surface(Beta, Alpha, f, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('$\\beta \\,(^\\circ)$', labelpad=10)
xmin, xmax = Beta.min(), Beta.max()
ax.set_xticks(np.round(np.linspace(xmin, xmax, 3)).astype("int"))
ax.set_ylabel('$\\alpha\' \\,(^\\circ)$', labelpad=10)
ymin, ymax = Alpha.min(), Alpha.max()
ax.set_yticks(np.linspace(ymin, ymax, 5))
ax.set_zlabel('f$(\\beta, \\alpha\')$')
ax.set_title(f'Reconstructed Intensity $f(\\beta, \\alpha\')$, \
    \n $C_f: {opt_params[0]:.2f}, \\sigma_f: {opt_params[1]:.2f}, \\beta_{{offset}}: {opt_params[2]:.2f}$')

# Colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Amplitude')

# Adjust view angle
ax.view_init(elev=45, azim=30)  # Elevation and azimuthal angles

plt.tight_layout()
plt.savefig("reconstructed-f.png", dpi=300, bbox_inches=None)
#plt.show()

alpha_prime_deg = np.rad2deg(alpha_prime)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(alpha_prime_deg, I, linewidth=2, linestyle='-', color='r', label='Intensity I')
# Add labels and title
ax.set_xlabel('$\\alpha\' \\,(^\\circ)$')
ax.set_ylabel('Intensity')
ax.set_title(f'Reconstructed Source Intensity I$(\\alpha, \\alpha\')$ \n $\\sigma_I: {sigmaI:.2f}$')
xmin, xmax = alpha_prime_deg.min(), alpha_prime_deg.max()
ax.set_xticks(np.linspace(xmin, xmax, 5))
plt.legend(loc=(0.01, 0.7))
plt.grid(True)
plt.savefig("reconstructed-I.png", dpi=300, bbox_inches="tight")
plt.show()
