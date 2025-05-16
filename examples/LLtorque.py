# This script precesses a single spin around a magnetic field without damping

import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util.constants import GAMMALL


def analytical(t, Bz):
    return np.cos(Bz * GAMMALL * t)


length, width, thickness = 1e-9, 1e-9, 1e-9
nx, ny, nz = 1, 1, 1  # single cell
world = World(cellsize=(length / nx, width / ny, thickness / nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)))
magnet.msat = 800e3

magnet.magnetization = (1, 0, 0)
Bz = 0.1
world.bias_magnetic_field = (0, 0, Bz)

tmax = 2e-9
timepoints = np.linspace(0, tmax, 75)
outputquantities = {"mx": lambda: magnet.magnetization.average()[0]}
output = world.timesolver.solve(timepoints, outputquantities)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 4.8))

time_HD = np.linspace(0, tmax, 1001)
ax.plot(1e9*time_HD, analytical(time_HD, Bz), 'k--', label="Analytical")
ax.plot(1e9*np.asarray(output["time"]), output["mx"], 'o', label=r"mumax⁺")

ax.set_title("Single spin precessing without damping")
ax.set_xlabel(r"time $t$ (ns)")
ax.set_ylabel(r"$m_x$")
ax.set_xlim(0, tmax*1e9)
ax.legend(loc="upper right")

plt.show()
