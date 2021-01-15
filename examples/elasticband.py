#!/bin/env python3

from mumax5 import *
from mumax5.util import *

import matplotlib.pyplot as plt
import numpy as np


def generate_images(grid, n):
    arrayshape = (3, grid.size[2], grid.size[1], grid.size[0])
    images = []
    for i in range(n):
        m = np.zeros(arrayshape)
        m[0] = np.cos(np.pi*i/(n-1))  # add small number to break symmetry
        m[1] = np.sin(np.pi*i/(n-1))
        images.append(m)

    images[ 0][1] += 0.1
    images[-1][1] += 0.1
    return images



world = World((3e-9, 3e-9, 3e-9))
magnet = Ferromagnet(world, Grid((64, 32, 1)))
magnet.msat = 800e3
magnet.aex = 13e-12


n_images = 10
eb = ElasticBand(magnet, generate_images(magnet.grid, n_images))
eb.relax_endpoints()

def show_energy_band(eb):
    energies=[]
    for i in range(n_images):
        eb.select_image(i)
        energies.append(magnet.total_energy.eval())

    def on_pick(event):
        plt.figure()
        eb.select_image(event.ind)
        show_field(magnet.magnetization)

    fig, ax = plt.subplots()
    ax.plot(energies,'o-',picker=10)
    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()

def plot_energy_band():
    energies=[]
    for i in range(n_images):
        eb.select_image(i)
        energies.append(magnet.total_energy.eval())
    plt.plot(energies)

for i in range(n_images-1):
    print(eb.geodesic_distance_images(i,i+1))

for i in range(1000):
    if i%100 == 0:
        plot_energy_band()
    eb.step(1.0e-3)

plt.show()

show_energy_band(eb)

#eb.select_image(0)
#show_field(magnet.magnetization)
