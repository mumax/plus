import numpy as np

from mumax5 import Ferromagnet, Grid, World
from mumax5.util import neelskyrmion, show_field


# NUMERICAL PARAMETERS RELEVANT FOR THE SPECTRUM ANALYSIS
fmax = 50E9           # maximum frequency (in Hz) of the sinc pulse
T = 2E-9              # simulation time (longer -> better frequency resolution)
dt = 1 / ( 2 * fmax)  # the sample time (Nyquist theorem taken into account)
t0 = 1 / fmax
d = 100E-9            # circle diameter
nx = 32               # number of cells


# create the world
grid_size = (nx, nx, 1)
cell_size = (d / nx, d / nx, 1E-9)

world = World(cell_size)

# create the ferromagnet
geometry_func = lambda x, y, z: x ** 2 + y ** 2 < (d / 2) ** 2
magnet = Ferromagnet(world, Grid(size=grid_size), geometry=geometry_func)
magnet.enable_demag = False
magnet.msat = 1E6
magnet.aex = 15E-12
# magnet.ku1 = lambda t: 1E6 * (1 + 0.01 * np.sinc(2 * fmax * (t - t0)))
magnet.idmi = 3E-3
magnet.anisU = (0, 0, 1)
magnet.alpha = 0.001

# set and relax the initial magnetization
magnet.magnetization = neelskyrmion(position=(0.5 * d, 0.5 * d, 0),
                                    radius=0.5 * d,
                                    charge=-1,
                                    polarization=1)
magnet.minimize()

world.timesolver.run(T)

show_field(magnet.magnetization)
