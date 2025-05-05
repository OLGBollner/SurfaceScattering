import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

intensity = readData("Source Scattering alpha_prime") # .5 deg step
intensity2 = readData("Source Scattering alpha_prime2") # .1 deg step

plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['legend.fontsize'] = 'xx-large'
plt.rcParams['axes.labelsize'] = 'xx-large'
plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rcParams['ytick.labelsize'] = 'xx-large'


plt.plot(intensity[:, 0], intensity[:, 1], linewidth=2, linestyle='-', color='b', label='Intensity 0.5 deg step')

# Add labels and title
plt.xlabel('Beta (deg)')
plt.ylabel('Intensity')
plt.title('Measured Average Intensity \n over wavelengths 150 - 1200 nm')
plt.legend(loc=(0.01, 0.7))
plt.grid(True)
plt.xlim(175, 185)
plt.ylim(2000, 7000)
plt.savefig("measured-intensity-05step-.png", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(intensity2[:, 0], intensity2[:, 1], linewidth=2, linestyle='-', color='r', label='Intensity 0.1 deg step')
# Add labels and title
plt.xlabel('Beta (deg)')
plt.ylabel('Intensity')
plt.title('Measured Average Intensity \n over wavelengths 150 - 1200 nm')
plt.legend(loc=(0.01, 0.7))
plt.grid(True)
plt.xlim(175, 185)
plt.ylim(2000, 7000)
plt.savefig("measured-intensity-01step-.png", dpi=300, bbox_inches="tight")
plt.show()
