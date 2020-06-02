from mumax5.engine import *
from mumax5.util import *

import numpy as np


def vortex(gridsize):
    """ returns magnetization configuration with a vortex at the center """
    m = np.zeros((3, gridsize[2], gridsize[1], gridsize[0]))
    x = np.linspace(-gridsize[0]/2., gridsize[0]/2., gridsize[0])
    y = np.linspace(-gridsize[1]/2., gridsize[1]/2., gridsize[1])
    z = np.linspace(-gridsize[2]/2., gridsize[2]/2., gridsize[2])
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    m[0] = -yy / (xx**2 + yy**2)
    m[1] = xx / (xx**2 + yy**2)
    return m


world = World(cellsize=(3e-9, 3e-9, 3e-9))

gridsize = (32, 64, 1)
offset = 34
n_magnets = 4

magnets = []
for i in range(n_magnets):
    grid = Grid(gridsize, origin=(i*offset, 0, 0))
    magnet = world.add_ferromagnet(grid, name=f"magnet_{i}")
    magnet.aex = 13e-12
    magnet.alpha = 0.5
    magnet.msat = 800e3
    magnet.magnetization = (1, 0.1, 0)
    magnets.append(magnet)

LLGequations = [(magnet.magnetization, magnet.torque) for magnet in magnets]

solver = TimeSolver(LLGequations)
solver.run(1e-10)

for magnet in magnets:
    show_field(magnet.magnetization)