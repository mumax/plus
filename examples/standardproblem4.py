#!/bin/env python3

# This script solves micromagnetic standard problem 4. The Problem specification
# can be found on https://www.ctcms.nist.gov/~rdm/mumag.org.html

import matplotlib.pyplot as plt
import numpy as np

from mumax5 import Ferromagnet, Grid, World

length, width, thickness = 500e-9, 125e-9, 3e-9
nx, ny, nz = 128, 32, 1

world = World(cellsize=(length / nx, width / ny, thickness / nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)))
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.02

magnet.magnetization = (1, 0.1, 0)
magnet.minimize()

B1 = (-24.6e-3, 4.3e-3, 0)
B2 = (-35.5e-3, -6.3e-3, 0)
world.bias_magnetic_field = B1  # choose B1 or B2 here

# --- SCHEDULE THE OUTPUT ---

timepoints = np.linspace(0, 1e-9, 1000)
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

output = world.timesolver.solve(timepoints, outputquantities)

# --- PLOT THE OUTPUT DATA ---

plt.subplot(211)
for key in ["mx", "my", "mz"]:
    plt.plot(output["time"], output[key], label=key)
plt.legend()

plt.subplot(212)
for key in ["e_total", "e_exchange", "e_zeeman", "e_demag"]:
    plt.plot(timepoints, output[key], label=key)
plt.legend()

plt.show()
