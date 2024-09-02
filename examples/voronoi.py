"""This script shows how to use Mumax+'s Voronoi Tesselator.
This is inspired by figure 19 of the paper "The design and
verification of MuMax3". https://doi.org/10.1063/1.4899186 """

import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import Circle, VoronoiTesselator
from mumaxplus.util import show_field, vortex

# Set up simulation parameters
N = 256
c = 4e-9
d = 40e-9

world = World(cellsize=(c, c, d))
grid = Grid((N, N, 1))

# Create a circle
diam = N*c
geo = Circle(diam).translate(diam/2, diam/2, 0)

# Initialize a Voronoi Tesselator using a grainsize of 40e-9 m
tesselator = VoronoiTesselator(world, grid, 40e-9)
regions = tesselator.generate()

# Create Ferromagnet
magnet = Ferromagnet(world, grid, geometry=geo, regions=regions)
magnet.alpha = 3
magnet.aex = 13e-12
magnet.msat = 860e3

# Set vortex magnetization
magnet.magnetization = vortex(magnet.center, 2*c, 1, 1)
show_field(magnet.magnetization)

for i in np.ravel(regions):
    # Set random anisotropy axes in each region
    anisC1 = (np.random.normal(), np.random.normal(), np.random.normal())
    anisC2 = (np.random.normal(), np.random.normal(), np.random.normal())
    magnet.anisC1.set_in_region(i, anisC1)
    magnet.anisC2.set_in_region(i, anisC2)

    # Create a random 10% anisotropy variation
    K = 1e5
    magnet.kc1.set_in_region(i, K + np.random.normal() * 0.1 * K)

    # TODO: vary interregion aex

# Evolve the world in time
world.timesolver.run(0.1e-9)

show_field(magnet.magnetization)

fig = plt.figure()
plt.imshow(magnet.kc1()[0][0], vmin=np.unique(magnet.kc1().flatten())[1],
            vmax=np.max(magnet.kc1()), cmap="Greys")
plt.title(r"First cubic anisotropy constant $K_{c1}$ (J / m³)")
plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$y$ (m)")
plt.colorbar()