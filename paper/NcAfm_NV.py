"""
This script generates and plots the data from Fig. 3 of the paper
'mumax+: extensible GPU-accelerated micromagnetics and beyond' (https://arxiv.org/abs/2411.18194).

The output will differ slightly due to a different random starting configuration.

Simulation takes about 5 minutes to complete.
"""
import numpy as np
import matplotlib.pyplot as plt

from mumaxplus import NcAfm, Grid, World, StrayField
from mumaxplus.util import VoronoiTessellator

nx, ny, nz = 502, 502, 1
world = World(cellsize=(1e-9, 1e-9, 30e-9))
grid = Grid((nx, ny, nz))

tess = VoronoiTessellator(40e-9, seed=1234567)
reg = tess.generate(world, grid)

magnet = NcAfm(world, grid, regions=reg)
magnet.msat = 1.0e6
magnet.alpha = 0.01

for i in np.unique(np.ravel(reg)).astype(int):
    if (i % 2 != 0):
        # in-plane
        magnet.anisU = (np.random.rand(), np.random.rand(), 0)
    else:
        # out-of-plane
        magnet.anisU = (0, 0, 1)

magnet.ku1 = 5e6

magnet.aex = 10e-12
magnet.ncafmex_cell = -25e-12
magnet.ncafmex_nn = -15e-12

magnet.scale_ncafmex_nn = 0.1
magnet.scale_exchange = 0.1

magnet.dmi_tensors.set_interfacial_dmi(1e7 * 0.35e-9)
magnet.dmi_vector = (0, 0, 1e7)

magnet.relax()

stray_field = StrayField(magnet, Grid((nx, ny, nz), origin=(0, 0, 2)))
data = stray_field()[2, 0, :, :] * 10000 # z-component in Gauss

fig, axs = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
im1 = axs.imshow(data, cmap="seismic", origin="lower", vmin=-6, vmax=6)
axs.set_title("Tip height: 60 nm")
axs.set_xlabel('x (nm)')
axs.set_ylabel('y (nm)')
cbar1 = fig.colorbar(im1, ax=axs, fraction=0.046, pad=0.04, aspect=30)
cbar1.ax.set_title(r'$B_z$ (G)', fontsize=10)
plt.show()