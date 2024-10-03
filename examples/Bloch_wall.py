"""This script computes the transition of a Bloch wall into a Neel wall
by varying the interfacially induced DMI parameter. This is a recreation
of figure 6 of the paper "The design and verification of MuMax3".
https://doi.org/10.1063/1.4899186 """

import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import twodomain

def line(x, a, b):
    return a * x + b

# --- Create world and add a Ferromagnet --

length, width, thickness = 250e-9, 250e-9, 0.6e-9
nx, ny, nz = 128, 128, 1
world = World(cellsize=(length / nx, width / ny, thickness / nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)))
magnet.msat = 1100e3
magnet.aex = 16e-12
magnet.alpha = 3

magnet.ku1 = 1.27e6
magnet.anisU = (0, 0, 1)

# --- Create a Bloch wall ---
magnet.magnetization = twodomain((0, 0, -1), (1, 1, 0), (0, 0, 1), length/2, 3e-9)

# --- Set up simulation parameters ---
mx, my = [], []

Dsteps = 50
Dmax = 0.3
D = np.linspace(0, Dmax, Dsteps)

# --- Vary DMI parameter and store magnetization and energy ---
for d in D:

    magnet.dmi_tensor.set_interfacial_dmi(d * 1e-3)
    magnet.minimize(tol=1e-5)

    mag = magnet.magnetization.average()
    mx.append(mag[0])
    my.append(mag[1])

fig = plt.figure(figsize=(7,5))
plt.plot(D, mx, 'o', label=r"$<m_x>$")
plt.plot(D, my, 'o', label=r"$<m_y>$")
plt.xlim(0, Dmax)
plt.ylim(-0.001, 0.1)
plt.xlabel(r"$D_{int}$ (mJ/m²)")
plt.ylabel("Domain wall moments (a.u.)")
plt.legend()
plt.savefig("bloch.png")