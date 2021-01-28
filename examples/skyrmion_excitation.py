import matplotlib.pyplot as plt
import numpy as np

from mumax5 import Ferromagnet, Grid, World
from mumax5.util import neelskyrmion, show_field


# NUMERICAL PARAMETERS RELEVANT FOR THE SPECTRUM ANALYSIS
fmax = 50E9           # maximum frequency (in Hz) of the sinc pulse
T = 2E-9              # simulation time (longer -> better frequency resolution)
dt = 1 / (2 * fmax)  # the sample time (Nyquist theorem taken into account)
t0 = 1 / fmax
d = 100E-9            # circle diameter
nx = 32               # number of cells


# create the world
grid_size = (nx, nx, 1)
cell_size = (d / nx, d / nx, 1E-9)

world = World(cell_size)

# create the ferromagnet
geometry_func = lambda x, y, z: (x - 4.84E-8) ** 2 +  (y -  4.84E-8) ** 2 <= (0.5 * d) ** 2

# mask = mask.reshape((1, 1, 32, 32)) # the bug is not cause by our mask, so its good
# magnet = Ferromagnet(world, Grid(size=grid_size), geometry=geometry_func)
magnet = Ferromagnet(world, Grid(size=grid_size))
magnet.geometry
magnet.msat = 1E6
magnet.aex = 15E-12

mask = np.zeros(shape=(1, 1, 32, 32))
magnet.ku1 = lambda t: 1E6 * (1 + 0.01 * np.sinc(2 * fmax * (t - t0))), mask
# magnet.ku1 = 0

magnet.idmi = 3E-3
magnet.anisU = (0, 0, 1)
magnet.alpha = 0.001

# set and relax the initial magnetization
magnet.magnetization = neelskyrmion(position=magnet.center,
                                    radius=0.5 * d,
                                    charge=-1,
                                    polarization=1)
magnet.minimize()

timepoints = np.linspace(0, T, int(T / dt))
outputquantities = {
    "mx": lambda: magnet.magnetization.average()[0],
    "my": lambda: magnet.magnetization.average()[1],
    "mz": lambda: magnet.magnetization.average()[2],
    "e_total": magnet.total_energy,
    "e_exchange": magnet.exchange_energy,
    "e_zeeman": magnet.zeeman_energy,
    "e_demag": magnet.demag_energy,
}

# --- RUN THE SOLVER ---

world.timesolver.timestep

output = world.timesolver.solve(timepoints, outputquantities)

# --- PLOT THE OUTPUT DATA ---

plt.figure(figsize=(10, 8))
for key in ["mx", "my", "mz"]:
    plt.plot(output["time"], output[key], label=key)
plt.legend()
plt.title("Mumax5 - Python")
plt.show()

# FAST FOURIER TRANSFORM
dm     = np.array(output["mz"]) - output["mz"][0]   # average magnetization deviaton
spectr = np.abs(np.fft.fft(dm))         # the absolute value of the FFT of dm
freq   = np.linspace(0, 1/dt, len(dm))  # the frequencies for this FFT

# PLOT THE SPECTRUM
plt.plot(freq/1e9, spectr)
plt.xlim(0,fmax/1e9)
plt.ylabel("Spectrum (a.u.)")
plt.xlabel("Frequency (GHz)")
plt.title("Mumax5 - Python")
plt.show()

show_field(magnet.magnetization)


plt.subplot(211)
for key in ["mx", "my", "mz"]:
    plt.plot(output["time"], output[key], label=key)
plt.legend()

plt.subplot(212)
for key in ["e_total", "e_exchange", "e_zeeman", "e_demag"]:
    plt.plot(timepoints, output[key], label=key)
plt.legend()

plt.show()
