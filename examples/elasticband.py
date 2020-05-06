#!/bin/env python3

from mumax5.engine import *
from mumax5.util import *

import matplotlib.pyplot as plt
import numpy as np


def generate_images(grid, n):
    arrayshape = (3, grid.size[2], grid.size[1], grid.size[0])
    images = []
    for i in range(n):
        m = np.zeros(arrayshape)
        m[0] = np.cos(np.pi*i/(n-1)+0.1)  # add small number to break symmetry
        m[1] = np.sin(np.pi*i/(n-1)+0.1)
        images.append(m)
    return images


world = World((3e-9, 3e-9, 3e-9))
magnet = world.add_ferromagnet(Grid((32, 16, 1)))
magnet.msat = 800e3
magnet.aex = 13e-12

n_images = 16
eb = ElasticBand(magnet, generate_images(magnet.grid, n_images))
eb.relax_endpoints()


energies_initial = []
for i in range(n_images):
    eb.select_image(i)
    energies_initial.append(magnet.total_energy.eval())

for i in range(4000):
    eb.step(1.0e-3)

energies_final = []
for i in range(n_images):
    eb.select_image(i)
    energies_final.append(magnet.total_energy.eval())

for i in range(n_images):
    eb.select_image(i)
    show_field(magnet.magnetization)

plt.plot(energies_initial)
plt.plot(energies_final)
plt.show()
