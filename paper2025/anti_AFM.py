"""
This script generates the data from Fig. 1 of the paper
'mumax+: extensible GPU-accelerated micromagnetics and beyond' (https://arxiv.org/abs/2411.18194)
and uses the mumax+ version v1.1.0.

Each datapoint in this figure is the result of one simulation using this script
with varying `(nc)afmex_cell`

The script below generates data for an Antiferromagnet. The results for a non-collinear
antiferromagnet or a ferromagnet can be obtained in an analogous way.
"""

import numpy as np

from mumaxplus import Antiferromagnet, Grid, World

# Cubic volumetric equivalent of spherical particle with a radius of 10 nm.
radius = 10e-9
V = (4/3) * np.pi * radius**3
size = V**(1/3)

N = 128
nx, ny, nz = N, N, 1
grid = Grid((nx, ny, nz))
world = World(cellsize=(size, size, size))

magnet = Antiferromagnet(world, grid)
magnet.msat = 400e3
magnet.alpha = 0.01

magnet.afmex_cell, power = 1e-15, 15 # gets varied

##############################
# No interaction between cells:
magnet.enable_demag = False
magnet.enable_openbc = True
##############################

magnet.anisU = (0, 0, 1)
magnet.ku1 = 2500

magnet.sub1.magnetization = (0, 0, 1)
magnet.sub2.magnetization = (0, 0, 1)

magnet.temperature = 400

tmax = 1e-8
dt = .1e-9 # sample every dt

output_file = "anti_AFM_A{}.txt".format(power)

timepoints = np.linspace(0, tmax, int(tmax / dt))
outputquantities = {"m1": lambda: magnet.sub1.magnetization.average()[2]}
output = world.timesolver.solve(timepoints, outputquantities, file_name=output_file)

# The switching time \tau is then extracted from this exponential decay by fitting
# a model of the form a * exp(-x / \tau).