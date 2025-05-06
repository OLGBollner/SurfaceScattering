import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import least_squares

def readData(directorypath):


    directory = Path(directorypath)

    intensity = np.empty((0,2))
    for file in directory.iterdir():
        angle = file.name[file.name.index("_")+1:file.name.index(".txt")]

        data = np.loadtxt(file, comments="#", usecols=1)
        meanIntensity = np.mean(data)

        intensity = np.vstack([intensity,
                               np.array([angle, meanIntensity])])

    intensity = intensity[intensity[:,0].argsort()].astype("float")

    return intensity

def estimateSourceParams():
    diameter = np.array([0.075, 0.065, 0.052, 0.045, 0.035])
    length = np.array([0.965, 0.933, 0.800, 0.720, 0.625])

    maxDeviation = np.mean(np.arctan((diameter/2) / length))
    sigmaI = maxDeviation/3

    return maxDeviation, sigmaI

def generateSource(alpha, deltaAlphaP):
    _, sigmaI = estimateSourceParams()
    alpha_prime = np.linspace(-np.pi/2, np.pi/2, round(1/deltaAlphaP))

    I = norm.pdf(alpha_prime, loc=alpha, scale=sigmaI)

    return I, alpha_prime

def generateCaptor(beta, deltaBetaP, sigmaf, C):
    beta_prime = np.linspace(-np.pi/2, np.pi/2, round(1/deltaBetaP))
    f = C * norm.pdf(beta[:, None] - beta_prime, sigmaf)

    return f

def residuals(params, I, beta, deltaAlphaP, MP):
    C, sigmaf = params

    f_init = generateCaptor(beta, deltaAlphaP, sigmaf, C)

    MP_pred = I @ f_init.T

    return MP_pred - MP

def estimateCaptor(deltaAlphaP, I):
    data = readData("Source Scattering alpha_prime2")

    beta = np.deg2rad(data[:, 0])
    MP = data[:, 1]
    
    initial_sigmaf, initial_C = 0.5, 0.5

    C_opt, sigmaf_opt = least_squares(
        residuals,
        [initial_C, initial_sigmaf],
        args=(I, beta, deltaAlphaP, MP)
    ).x

    f = generateCaptor(beta, deltaAlphaP, sigmaf_opt, C_opt)

    return f, C_opt, sigmaf_opt



intensity = readData("Source Scattering alpha_prime2") # .1 deg step

plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['legend.fontsize'] = 'xx-large'
plt.rcParams['axes.labelsize'] = 'xx-large'
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rcParams['ytick.labelsize'] = 'xx-large'


plt.plot(intensity[:, 0], intensity[:, 1], linewidth=2, linestyle='-', color='r', label='Intensity 0.1 deg step')
# Add labels and title
plt.xlabel('β (deg)')
plt.ylabel('Intensity')
plt.title('Measured Average Intensity \n over wavelengths 150 - 1200 nm')
plt.legend(loc=(0.01, 0.7))
plt.grid(True)
plt.xlim(175, 185)
plt.ylim(2000, 7000)
plt.savefig("measured-intensity-01step.png", dpi=300, bbox_inches="tight")
plt.show()


deltaAlphaP = 0.001
alpha = np.array([0])
beta = np.deg2rad(intensity[:, 0])

I, alpha_prime = generateSource(alpha, deltaAlphaP)
f, C_opt, sigmaf_opt = estimateCaptor(deltaAlphaP, I)

print(f"C: {C_opt}, Sigma_f: {sigmaf_opt}")

# Create a meshgrid for 3D plotting
Beta, Alpha = np.rad2deg(np.meshgrid(beta, alpha_prime, indexing='ij'))

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type('ortho')

# Surface plot
surf = ax.plot_surface(Beta, Alpha, f, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('β (deg)', fontsize=12)
ax.set_ylabel('α\' (deg)', fontsize=12)
ax.set_zlabel('f(β, α\')', fontsize=12)
ax.set_title('3D Gaussian Kernel $f(β, α\')$', fontsize=14)

# Colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Amplitude')

# Adjust view angle
ax.view_init(elev=30, azim=45)  # Elevation and azimuthal angles

plt.tight_layout()
plt.savefig("reconstructed-f.png", dpi=300, bbox_inches="tight")
plt.show()

# plt.figure(figsize=(8, 6))
# plt.imshow(f, extent=np.rad2deg([alpha_prime.min(), alpha_prime.max(), beta.min(), beta.max()]),
#            origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=np.max(f))
# plt.colorbar(label='f(β, α\')')
# plt.xlabel('α\' (deg)')
# plt.ylabel('β (deg)')
# plt.title('Gaussian Kernel f(β, α\')')
# plt.savefig("reconstructed-f.png", dpi=300, bbox_inches="tight")
# plt.show()

alpha_prime_deg = np.rad2deg(alpha_prime)
plt.plot(alpha_prime_deg, I, linewidth=2, linestyle='-', color='r', label='Intensity I')
# Add labels and title
plt.xlabel('α\' (deg)')
plt.ylabel('Intensity')
plt.title('Reconstructed Source intensity')
plt.legend(loc=(0.01, 0.7))
plt.xlim(alpha_prime_deg.min(), alpha_prime_deg.max())
plt.grid(True)
plt.savefig("reconstructed-I.png", dpi=300, bbox_inches="tight")
plt.show()
