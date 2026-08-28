"""This test is based on the 1D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c
   It compares the final magnetization of both sublattices in an antiferromagnet
   with that of a ferromagnet. All AFM exchanges are set to 0 for this test."""

import numpy as np
from mumaxplus import Ferromagnet, Antiferromagnet, Grid, World

RTOL = 1e-5
def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

def simulations(openbc, interfacial):
    """This simulates a 1D wire with bulk or interfacial DMI in both
       a ferromagnet and an antiferromagnet."""

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
    
    # antiferromagnet simulation
    world_AFM = World(cellsize=cellsize)
    magnet_AFM = Antiferromagnet(world_AFM, Grid(gridsize))
    magnet_AFM.enable_demag = False
    magnet_AFM.enable_openbc = openbc

    magnet_AFM.msat = Ms
    magnet_AFM.aex = A
    magnet_AFM.afmex_cell = 0
    magnet_AFM.afmex_nn = 0
    magnet_AFM.ku1 = Ku
    magnet_AFM.anisU = anisU

    for sub in magnet_AFM.sublattices:
        if interfacial:
            sub.dmi_tensor.set_interfacial_dmi(D)
        else:
            sub.dmi_tensor.set_bulk_dmi(D)
    
    magnet_AFM.magnetization = magnetization

    magnet_AFM.minimize()

    return  magnet, magnet_AFM


class TestDMI1D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test_closed_inter(self):
        magnet, magnet_AFM = simulations(False, True)
        err = max_relative_error(magnet.magnetization.eval(), magnet_AFM.sub1.magnetization.eval())
        assert err < RTOL
        err = max_relative_error(magnet.magnetization.eval(), magnet_AFM.sub2.magnetization.eval())
        assert err < RTOL

    def test_closed_bulk(self):
        magnet, magnet_AFM = simulations(False, False)
        err = max_relative_error(magnet.magnetization.eval(), magnet_AFM.sub1.magnetization.eval())
        assert err < RTOL
        err = max_relative_error(magnet.magnetization.eval(), magnet_AFM.sub2.magnetization.eval())
        assert err < RTOL

    def test_open_inter(self):
        magnet, magnet_AFM = simulations(True, True)
        err = max_relative_error(magnet.magnetization.eval(), magnet_AFM.sub1.magnetization.eval())
        assert err < RTOL
        err = max_relative_error(magnet.magnetization.eval(), magnet_AFM.sub2.magnetization.eval())
        assert err < RTOL

    def test_open_bulk(self):
        magnet, magnet_AFM = simulations(True, False)
        err = max_relative_error(magnet.magnetization.eval(), magnet_AFM.sub1.magnetization.eval())
        assert err < RTOL
        err = max_relative_error(magnet.magnetization.eval(), magnet_AFM.sub2.magnetization.eval())
        assert err < RTOL