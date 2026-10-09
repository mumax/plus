# This script shows the uniaxial and cubic anisotropy energies for multiple
# magnetization directions in the xy-plane.

import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World


def polar(uni, cub):
    zeros = (0, 0, 0)
    if uni:
        # reset unneeded cubic anisotropy parameters
        magnet.kc1, magnet.kc2, magnet.kc3 = zeros
        magnet.anisC1 = zeros
        magnet.anisC2 = zeros
        # add specified uniaxial anisotropy parameters
        magnet.ku1, magnet.ku2 = uni
        magnet.anisU = (1, 0, 0)
    
    if cub:
        # reset unneeded uniaxial anisotropy parameters
        magnet.ku1, magnet.ku2 = 0, 0
        magnet.anisU = zeros
        # add cubic uniaxial anisotropy parameters
        magnet.kc1, magnet.kc2, magnet.kc3 = cub
        magnet.anisC1 = (1, 0, 0)
        magnet.anisC2 = (0, 1, 0)
        
    angles = np.linspace(0, 2 * np.pi, 50)  # sample some magn directions
    energies = []
    energies_theo = []

    for th in angles:
        # set uniform magnetiation in xy-plane for specific direction
        mx, my, mz = np.cos(th), np.sin(th), 0
        magnet.magnetization = (mx, my, mz)
        # save Edens_anisotropy for the only cell
        energies.append(magnet.anisotropy_energy_density.eval()[0,0,0,0])

        # save theoretical anisotropy energy density
        if uni:  # uniaxial
            E_theo = (uni[0] * (-mx ** 2) - uni[1] * (mx ** 4))
        if cub:  # cubic
            E_theo = (cub[0] * (mx ** 2) * (my ** 2) + cub[2] * (mx ** 4) * (my ** 4))
        energies_theo.append(E_theo)

    if uni:  # uniaxial plot labels, linestyles and ticks
        lab = "Ku_1" if uni[0] else "Ku_2"
        s = '-' if uni[0]<0 or uni[1]<0 else ''
        rticks = [-1000, 0, 1000]

    elif cub:  # cubic plot labels, linestyles and ticks
        lab = "Kc_1" if cub[0] else "Kc_3"
        s = '-' if cub[0]<0 or cub[2]<0 else ''
        rticks = [-250, 0, 250]

    plt.polar(angles, energies_theo, 'k--')
    plt.polar(angles, energies, 'o', label=r"${} = ${}1 kJ/m³".format(lab, s))
    ax.set_rticks(rticks)
    ax.set_rlabel_position(0)
    ax.set_xticklabels([])
    plt.legend()


world = World(cellsize=(1e-9, 1e-9, 1e-9))
magnet = Ferromagnet(world, Grid((1, 1, 1)))

fig = plt.figure(figsize=(7,7))
# uniaxial anisotropy
ax = fig.add_subplot(221, projection="polar")
polar([1e3, 0], [])
polar([-1e3, 0], [])
ax = fig.add_subplot(222, projection="polar")
polar([0, 1e3], [])
polar([0, -1e3], [])

# cubic anisotropy
ax = fig.add_subplot(223, projection="polar")
polar([], [1e3, 0, 0])
polar([], [-1e3, 0, 0])
ax = fig.add_subplot(224, projection="polar")
polar([], [0, 0, 1e3])
polar([], [0, 0, -1e3])
plt.tight_layout()

plt.show()
