#!/bin/env python3

from mumax5 import Ferromagnet, Grid, World
from mumax5.util import show_field

length, width, thickness = 500e-9, 125e-9, 3e-9
nx, ny, nz = 128, 32, 1

world = World(cellsize=(length / nx, width / ny, thickness / nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)))
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.02
magnet.temperature = 200

magnet.magnetization = (1, 0.1, 0)

world.timesolver.run(1e-10)

show_field(magnet.magnetization)
