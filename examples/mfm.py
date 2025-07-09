"""In this example we create 2 magnets and visualize them using
   magnetic force microscopy."""

import matplotlib.pyplot as plt

from mumaxplus import Ferromagnet, Grid, World
import mumaxplus.util.config as config
from mumaxplus.util import MFM

# define parameters
msat = 566e3
aex = 2.48e-12
alpha = 1e-3

# Create a simulation world
world = World(cellsize=(2e-9,2e-9,1e-9))

# Add a ferromagnet
magnet1 = Ferromagnet(world, Grid((50, 50, 1)))
magnet1.magnetization = config.vortex(magnet1.center, diameter=800e-9, circulation=1, polarization=1)
magnet1.msat = msat
magnet1.aex = aex
magnet1.alpha = alpha

# Add another ferromagnet
magnet2 = Ferromagnet(world, Grid((50, 50, 1), origin=(60,60,0)))
magnet2.magnetization = (1,0,0)
magnet2.msat = msat
magnet2.aex = aex
magnet2.alpha = alpha

print("Relaxing magnets...")
world.relax()

print("Creating MFM images...")

# Create an MFM instance for the enitre world
grid_world = Grid((120, 120, 1))
mfm_world= MFM(world, grid_world)

mfm_world.lift = 5e-9
world_image = mfm_world.eval()
plt.imshow(world_image[0,0,...], cmap="gray", origin="lower")
plt.xlabel("x (m)")
plt.xlabel("y (m)")
plt.title("MFM image of everything in the world")
plt.show()

# We can also only look at one magnet
grid_magnet = Grid((70, 70, 1), origin=(50, 50, 0))
mfm_magnet = MFM(magnet2, grid_magnet)
mfm_magnet.lift = 20e-9

magnet_image = mfm_magnet.eval()
plt.imshow(magnet_image[0,0,...], cmap="gray", origin="lower")
plt.xlabel("x (m)")
plt.xlabel("y (m)")
plt.title("MFM image of only one magnet")
plt.show()