from mumax5 import Ferromagnet, Grid, World
from mumax5.util import show_field

world = World(cellsize=(0.2, 0.2, 0.2))
magnet = Ferromagnet(world, Grid((128, 128, 1)))
magnet.enable_demag = False
magnet.aex = 1
magnet.ku1 = 1
magnet.anisU = (0, 0, 1)
magnet.idmi = 1.3
magnet.alpha = 0.8

world.timesolver.run(2e-10)

show_field(magnet.magnetization)
