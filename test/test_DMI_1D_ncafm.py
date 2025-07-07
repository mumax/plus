"""This test is based on the 1D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c
   It compares the final magnetization of all sublattices in a non-collinear
   antiferromagnet with that of a ferromagnet. All antiferromagnetic exchanges
   are set to 0 for this test."""

import pytest
import numpy as np
from mumaxplus import Ferromagnet, NcAfm, Grid, World

RTOL = 1e-5
def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

def simulations(openbc, interfacial):
    """This simulates a 1D wire with bulk or interfacial DMI in both
       a ferromagnet and a non-collinear antiferromagnet."""

    # constants
    A = 13e-12
    D = 3e-3
    Ku = 0.4e6
    anisU = (1,0,0)
    Ms = 0.86e6

    cellsize = (1e-9, 1e-9, 1e-9)
    gridsize = (1, 1, 100)
    magnetization = (1,0,0)

    # ferromagnet simulation
    world = World(cellsize=cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))
    magnet.enable_demag = False
    magnet.enable_openbc = openbc

    magnet.msat = Ms
    magnet.aex = A
    magnet.ku1 = Ku
    magnet.anisU = anisU

    if interfacial:
        magnet.dmi_tensor.set_interfacial_dmi(D)
    else:
        magnet.dmi_tensor.set_bulk_dmi(D)
    
    magnet.magnetization = magnetization
    magnet.minimize()
    
    # NcAfm simulation
    world_NcAfm = World(cellsize=cellsize)
    magnet_NcAfm = NcAfm(world_NcAfm, Grid(gridsize))
    magnet_NcAfm.enable_demag = False
    magnet_NcAfm.enable_openbc = openbc

    magnet_NcAfm.msat = Ms
    magnet_NcAfm.aex = A
    magnet_NcAfm.ncafmex_cell = 0
    magnet_NcAfm.ncafmex_nn = 0
    magnet_NcAfm.ku1 = Ku
    magnet_NcAfm.anisU = anisU

    for sub in magnet_NcAfm.sublattices:
        if interfacial:
            sub.dmi_tensor.set_interfacial_dmi(D)
        else:
            sub.dmi_tensor.set_bulk_dmi(D)
    
    magnet_NcAfm.magnetization = magnetization

    magnet_NcAfm.minimize()

    return  magnet, magnet_NcAfm


class TestDMI1D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test_closed_inter(self):
        magnet, magnet_NcAfm= simulations(False, True)
        for i in range(3):
            sub = magnet_NcAfm.sublattices[i]
            err = max_relative_error(magnet.magnetization.eval(), sub.magnetization.eval())
            assert err < RTOL

    def test_closed_bulk(self):
        magnet, magnet_NcAfm = simulations(False, False)
        for i in range(3):
            sub = magnet_NcAfm.sublattices[i]
            err = max_relative_error(magnet.magnetization.eval(), sub.magnetization.eval())
            assert err < RTOL

    def test_open_inter(self):
        magnet, magnet_NcAfm = simulations(True, True)
        for i in range(3):
            sub = magnet_NcAfm.sublattices[i]
            err = max_relative_error(magnet.magnetization.eval(), sub.magnetization.eval())
            assert err < RTOL


    def test_open_bulk(self):
        magnet, magnet_NcAfm = simulations(True, False)
        for i in range(3):
            sub = magnet_NcAfm.sublattices[i]
            err = max_relative_error(magnet.magnetization.eval(), sub.magnetization.eval())
            assert err < RTOL