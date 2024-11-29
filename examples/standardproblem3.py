# This script solves micromagnetic standard problem 3. The Problem specification
# can be found on https://www.ctcms.nist.gov/~rdm/mumag.org.html
# Only flower and vortex states are studied, but the mumax³ paper shows there to
# be a twisted flower state and a canted vortex state as well.
# https://doi.org/10.1063/1.4899186

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import vortex, show_field_3D

# A cube with edge length, L, expressed in units lex = (A/Km)1/2
# where Km is a magnetostatic energy density, Km = 1/2 µ0 Msat² (SI)
# So we express lengths in 'units' l_ex and energies in 'units' Km
# by setting both to 1.
mu0 = 1.25663706212e-6
msat = np.sqrt(2/mu0)  # so Km = 1/2 µ0 Msat² = 1
aex = 1.0  # so l_ex = sqrt(aex / Km) = 1

ku1 = 0.1  # 0.1 * Km as specified
anisU = (1, 0, 0)  # switched x and z for convenience

L_min, L_max, L_step = 8.0, 9.0, 0.1  # lengths in units of exchange length l_ex
L_array = np.arange(L_min, L_max + 0.5*L_step, L_step)

N = 32  # maximum cell length of L_max/N = 0.281 l_ex

def setup_magnet(L):
    world = World(cellsize=(L/N, L/N, L/N))
    magnet = Ferromagnet(world, Grid((N, N, N)))

    magnet.msat = msat
    magnet.aex = aex
    magnet.ku1 = ku1
    magnet.anisU = anisU

    return world, magnet

flower_E_tots = []
vortex_E_tots = []
for L in tqdm(L_array):
    world, magnet = setup_magnet(L)

    # flower
    magnet.magnetization = (1, 0, 0.01)  # right, with slight symmetry breaking    
    magnet.minimize()
    flower_E_tots.append(magnet.total_energy.eval())

    # vortex
    magnet.magnetization = vortex((L/2, L/2, L/2), L/12, -1, 1)
    magnet.minimize()
    vortex_E_tots.append(magnet.total_energy.eval())

# linearly interpolate cross-over point
E_diff = np.asarray(flower_E_tots) - np.asarray(vortex_E_tots)
i = np.where(E_diff > 0)[0][0]  # cross over index
L_cross = (L_array[i-1]*E_diff[i] - L_array[i]*E_diff[i-1]) / (E_diff[i] - E_diff[i-1])

# plot energies
fig, ax = plt.subplots()
ax.plot(L_array, flower_E_tots, marker="s", label="flower")
ax.plot(L_array, vortex_E_tots, marker="s", label="vortex")
ax.axvline(x=L_cross, ls="--", c="r")
ax.set_xlabel(r"$L/l_{\rm ex}$")
ax.set_ylabel(r"$E_{\rm tot}/K_{\rm m}$")
ax.legend()
plt.show()

# show states at phase transition
show_states = True
quiver = True  #  with(out) arrows
if show_states:
    world, magnet = setup_magnet(L_cross)

    # flower
    magnet.magnetization = (1, 0, 0.01)  # right, with slight symmetry breaking    
    magnet.minimize()
    show_field_3D(magnet.magnetization, quiver=quiver)

    # vortex
    magnet.magnetization = vortex((L/2, L/2, L/2), L/12, -1, 1)
    magnet.minimize()
    show_field_3D(magnet.magnetization, quiver=quiver)
