import numpy as np
import matplotlib.pyplot as plt


def plotIntensityHeatmap(distribution, angle, angleP, sigma, source):
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(np.rad2deg(angle), np.rad2deg(angleP), distribution, shading='auto')
    plt.colorbar(label=f'Intensity {source}')
    plt.xlabel('Source angle (degrees)')
    plt.ylabel('Ray angle (degrees)')
    plt.title(f'Gaussian Intensity Distribution\n(σ_{source} = {np.rad2deg(sigma)}°)')
    plt.show()

def plotIntensityAngles(distribution, angle, angleP, sigma, source):
    plt.figure(figsize=(10, 6))

    # Select some representative alpha values
    for alpha_val in range(0, np.size(angle), 33):
        plt.plot(np.rad2deg(angleP), distribution[alpha_val], 
                 label=f'angle={np.rad2deg(angle[alpha_val]):.0f}°')

    plt.xlabel('Ray angle (degrees)')
    plt.ylabel(f'Intensity {source}')
    plt.title(f'Gaussian Intensity for Different Angles \n(σ_{source} = {np.rad2deg(sigma)}°)')
    plt.legend()
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_gaussian_intensity_distribution(sigma_deg=5):
    """
    Recreates Figure III.2.I.5 showing Gaussian intensity distribution
    as a function of angular deviation from central angle.
    
    Parameters:
    - sigma_deg: Width parameter in degrees (default 5°)
    """
    # Convert to radians for calculations
    sigma = np.deg2rad(sigma_deg)
    
    # Create angular deviation range (-3σ to +3σ)
    theta = np.linspace(-3*sigma, 3*sigma, 200)
    theta_deg = np.rad2deg(theta)  # For plotting in degrees
    
    # Create Gaussian distribution
    intensity = norm.pdf(theta, 0, sigma)
    intensity /= intensity.max()  # Normalize to 1
    

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Main plot
    plt.plot(theta_deg, intensity, 'b-', linewidth=2, label=f'σ = {sigma_deg}°')
    
    # Highlight sigma regions
    plt.axvline(x=sigma_deg, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=-sigma_deg, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=2*sigma_deg, color='g', linestyle=':', alpha=0.5)
    plt.axvline(x=-2*sigma_deg, color='g', linestyle=':', alpha=0.5)
    
    # Fill between ±σ
    plt.fill_between(theta_deg, intensity, 
                    where=(theta_deg >= -sigma_deg) & (theta_deg <= sigma_deg),
                    color='blue', alpha=0.2)
    
    # Annotations
    plt.text(0, 0.6, '68% of rays', ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(sigma_deg*1.2, 0.3, f'σ = {sigma_deg}°', color='r')
    plt.text(-sigma_deg*1.2, 0.3, f'-σ = -{sigma_deg}°', color='r')
    
    # Formatting
    plt.title('Gaussian Intensity Distribution of Light Source/Captor\n'
             '(Figure III.2.I.5 Recreation)', pad=20)
    plt.xlabel('Angular deviation from central angle (degrees)')
    plt.ylabel('Normalized Intensity')
    plt.ylim(0, 1.1)
    plt.xlim(np.rad2deg(-np.pi/2), np.rad2deg(np.pi/2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return plt.gcf()

