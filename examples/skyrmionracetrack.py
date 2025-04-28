from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import neelskyrmion
import matplotlib.pyplot as plt
import numpy as np

# create the world
cellsize = (1e-9, 1e-9, 0.4e-9)
world = World(cellsize)

# create the ferromagnet
nx, ny, nz = 128, 64, 1
magnet = Ferromagnet(world, Grid(size=(nx, ny, nz)))
magnet.enable_demag = False
magnet.msat = 580e3
magnet.aex = 15e-12
magnet.ku1 = 0.8e6
magnet.anisU = (0, 0, 1)
magnet.alpha = 0.2
magnet.dmi_tensor.set_interfacial_dmi(3.2e-3)

# set and relax the initial magnetization
magnet.magnetization = neelskyrmion(
    position=(64e-9, 32e-9, 0), radius=5e-9, charge=-1, polarization=1
)

print("minimizing...")
magnet.minimize()
rgbs = [np.transpose(magnet.magnetization.get_rgb()[:,0,:,:], axes=[1,2,0])]
times = [world.timesolver.time]

# add a current
magnet.xi = 0.3
magnet.jcur = (1e12, 0, 0)
magnet.pol = 0.4

print("running...")
for i in range(2):
    world.timesolver.run(3e-10)
    rgbs.append(np.transpose(magnet.magnetization.get_rgb()[:,0,:,:], axes=[1,2,0]))
    times.append(world.timesolver.time)

# -------------------------
# plot
print("plotting...")

fig, axs = plt.subplots(nrows=3, sharex="all", sharey="all", figsize=(6, 8))

fig.suptitle("magnetization")

extent = [- 0.5 * cellsize[0] * 1e9, cellsize[0] * (nx - 0.5) * 1e9,
          - 0.5 * cellsize[1] * 1e9, cellsize[1] * (ny - 0.5) * 1e9]

for ax, rgb, time in zip(axs, rgbs, times):
    ax.imshow(rgb, origin="lower", extent=extent, )
    ax.set_title(f"t = {time*1e9:.2f} ns")
    ax.set_aspect("equal")

axs.flatten()[-1].set_xlabel("x (nm)")
axs.flatten()[1].set_ylabel("y (nm)")

fig.tight_layout()

plt.show()
