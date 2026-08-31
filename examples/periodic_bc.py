from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import get_rgb
import numpy as np
import matplotlib.pyplot as plt


nx, ny, nz = 128, 128, 1
cellsize = (0.1, 0.1, 0.1)

# infinite grid in z direction, periodic in x and y
mastergrid = Grid(size=(nx, ny, 0), origin=(0, 0, 0))  # not actually necessary
pbc_repetitions = (4, 4, 0)  # demagnetization repetitions

world = World(cellsize, pbc_repetitions=pbc_repetitions, mastergrid=mastergrid)
magnet = Ferromagnet(world, grid=Grid((nx, ny, nz)))

# this would automatically find a bounding mastergrid, but it would recalculate
# the demagnetization kernel, which could take a while for larger systens
# world.set_pbc(pbc_repetitions)

magnet.dmi_tensor.set_interfacial_dmi(1.5)
magnet.msat = 1.0
magnet.aex = 1.0
magnet.ku1 = 1.0
magnet.anisU = (0, 0, 1)
magnet.alpha = 0.5

world.timesolver.steps(500)

# --------------------------------------------------
# show magnetization 3x3 times to see no boundaries

rgb = get_rgb(magnet.magnetization, layer=0)  # 3x1x128x128 to 128x128x3
rgb = np.tile(rgb, (3, 3, 1))  # tile to 3x3

fig, ax = plt.subplots()
ax.imshow(rgb, origin="lower")

ticks = range(0, 3*nx, nx//4)
labels = [f"{(i%nx)*cellsize[0]:.0f}" for i in ticks]
ax.set_xticks(ticks, labels)
ax.set_yticks(ticks, labels)

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("3x3 magnetiation")

plt.show()
