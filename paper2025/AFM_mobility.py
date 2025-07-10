"""
This script generates the data from Fig. 1 of the paper
'mumax+: extensible GPU-accelerated micromagnetics and beyond' (https://arxiv.org/abs/2411.18194)
and uses the mumax+ version v1.1.0.

This simulation can take a significant time to complete (depending on your machine).
"""

import numpy as np

from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util import VoronoiTessellator


def get_domain_wall_speed(magnet):
    """Find stationary value of the velocity"""
    t = 5e-10
    neel = magnet.neel_vector()[0,:,:,:]
    pos1 = np.mean(np.argmin(np.abs(neel), axis=0))
    world.timesolver.run(t)
    neel = magnet.neel_vector()[0,:,:,:]
    pos2 = np.mean(np.argmin(np.abs(neel), axis=0))
    return (pos2 - pos1) * cz / t

def initialize(magnet):
    """Create a two-domain state"""
    nz2 = nz // 5
    dw2 = dw // 2
    m = np.zeros(magnet.sub1.magnetization.shape)
    m[0,         0:nz2 - dw2, :, :] = -1
    m[2, nz2 - dw2:nz2 + dw2, :, :] = 1  # Domain wall has a width of 4 nm.
    m[0, nz2 + dw2:         , :, :] = 1

    magnet.sub1.magnetization = m
    magnet.sub2.magnetization = -m
    magnet.relax()

cx, cy, cz = 10e-9, 2e-9, 2e-9  # Cellsize
nx, ny, nz = 1, 100, 400  # Number of cells
dw = 4  # Width of the domain wall in number of cells

world = World(cellsize=(cx, cy, cz))
grid = Grid((nx, ny, nz))

tess = VoronoiTessellator(10e-9)
regions = tess.generate(world, grid)

magnet = Antiferromagnet(world, grid, regions=regions)
magnet.msat = 0.4e6
magnet.alpha = 0.1

magnet.dmi_tensors.set_interfacial_dmi(0.11e-3)

magnet.ku1 = 64e3
magnet.anisU = (1, 0, 0)

magnet.aex = 10e-12
magnet.afmex_cell = -50e-12
magnet.afmex_nn = -10e-12

# Scale intergrain exchange (this gets varied)
magnet.sub1.scale_exchange = 0.25
magnet.sub2.scale_exchange = 0.25
magnet.scale_afmex_nn = 0.25

# Add current density
jrange = np.linspace(0, 4, 40)
magnet.pol = 0.044
magnet.fixed_layer = (0, 1, 0)
magnet.Lambda = 1
magnet.free_layer_thickness = cy

# collect data in list
speeds = []
for j in jrange:
    initialize(magnet)
    magnet.jcur = (0, 0, j * 1e12)
    s = get_domain_wall_speed(magnet)
    speeds.append(s)

np.save("AFM_mobility_speed_data", np.array(speeds))