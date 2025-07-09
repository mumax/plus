"""This test compares the amount of Néel vector
canting against the results from
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.103.134413

See also the discussion in `test_boundaries.py`.
"""

import pytest

import numpy as np
from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util import *

RTOL = 4e-2 # 4%

def canting(magnet):
    return magnet.neel_vector.eval()[2,...]/np.linalg.norm(magnet.neel_vector.eval(), axis=0)

def analytical(x, D):
    k = np.sqrt(1 - (2 * D / (np.pi * Dc))**2)
    return np.tanh(np.arctanh(k) + (N//2 - np.abs(x - N//2)))

@pytest.mark.slow
def test_neel_canting():
    a = 0.35e-9
    J = 2.34e-22
    H = 9.36e-23
    A = J / (2 * a)
    Ku = H / (2 * a**3)
    Ms = GAMMALL * HBAR / (2 * a**3)

    global N
    N = 32
    dx = 2*a

    world = World((dx, dx, dx))
    magnet = Antiferromagnet(world, Grid((N, N, N)))
    magnet.alpha = 0.5 / 2
    magnet.msat = Ms
    magnet.aex = A
    magnet.afmex_nn = -A
    magnet.anisU = (0, 0, 1)
    magnet.ku1 = Ku

    magnet.afmex_cell = - 6 * J / a**3 * HBAR

    magnet.sub1.magnetization = (0, 0, 1)
    magnet.sub2.magnetization = (0, 0, -1)

    magnet.enable_demag = False
    magnet.enable_openbc = False

    global Dc # Critical DMI
    Dc = 4 * np.sqrt(A/2 * Ku) / np.pi
    D = np.linspace(Dc / 10, Dc, 5)

    for d in D:
        magnet.dmi_tensors.set_bulk_dmi(d)
        magnet.relax()

        cant = canting(magnet)[N//2, N//2,:]
        assert np.abs(cant[0] - analytical(0, d/2)) / np.abs(analytical(0, d/2)) < RTOL