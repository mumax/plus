import pytest
import numpy as np

from mumaxplus import Ferromagnet, Grid, World
import mumaxplus.util.shape as shape

RTOL = 1e-3  # 0.1%

def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

def simulate():
    """Creates a mumax⁺ world with the same settings as the
    PBC1.mx3 test in mumax³ without PBC.    
    """
    c = 5e-9
    nx = 128
    ny = int(nx/2)

    # === mumax⁺ ===
    world = World(cellsize=(c,c,c))
    r = shape.Rectangle(nx/2*c, ny/2*c)
    r.translate((nx/2)*c, (ny/2)*c, 0)

    magnet = Ferromagnet(world, Grid((nx*5, ny*5, 1)), geometry=r.repeat((0, 0, None), ((nx)*c, (ny)*c, None)))
    magnet.msat = 1000e3
    magnet.aex = 10e-12
    magnet.alpha = 1
    magnet.magnetization = (1., .1, .01)

    return world, magnet


def simulate_PBC(corners=False):
    """Creates a mumax⁺ world with the same settings as the
    PBC1.mx3 test in mumax³ with PBC.    
    """
    c = 5e-9
    nx = 128
    ny = int(nx/2)

    # === mumax⁺ PBC ===
    world_PBC = World(cellsize=(c,c,c), mastergrid=Grid((nx,ny,0)), pbc_repetitions=(2,2,0))
    r_PBC = shape.Rectangle(nx/2*c, ny/2*c)
    r_PBC.translate((nx/2)*c, (ny/2)*c, 0)

    if not corners:
        magnet_PBC = Ferromagnet(world_PBC, Grid((nx, ny, 1)), geometry=r_PBC.repeat((0, 0, None), ((nx)*c, (ny)*c, None)))
    else:
        magnet_PBC = Ferromagnet(world_PBC, Grid((nx, ny, 1)), geometry=r_PBC.repeat((0, 0, None), ((nx)*c, (ny)*c, None)).translate(nx/2*c, ny/2*c, 0))
    magnet_PBC.msat = 1000e3
    magnet_PBC.aex = 10e-12
    magnet_PBC.alpha = 1
    magnet_PBC.magnetization = (1., .1, .01)

    return world_PBC, magnet_PBC


class TestPBC:
    def test_PBC_demag(self):
        """Check if the demagnetization fields of the central magnet in the simulation
        without PBC is the same as the one in the simulation with PBC.
        """
        nx = 128
        ny = int(nx/2)
        world_PBC, magnet_PBC = simulate_PBC()
        world, magnet = simulate()

        PBC_demag, demag = magnet_PBC.demag_field.eval(), magnet.demag_field.eval()
        err = max_relative_error(result=np.array(PBC_demag)[:,:, int(ny/4)+1:int(3*ny/4)+1, int(nx/4)+1:int(3*nx/4)+1],
                                 wanted=np.array(demag)[:,:, int(ny*5/2 - ny/4)+1:int(ny*5/2 + ny/4)+1, int(nx*5/2 - nx/4)+1:int(nx*5/2 + nx/4)+1])
        assert err < RTOL
    
    def test_PBC_run_center(self):
        """Perform the same comparison with the PBC as in mumax³ with the magnet in the center."""
        world_PBC, magnet_PBC = simulate_PBC()
        world_PBC.timesolver.run(1e-9)
        av_mag = magnet_PBC.magnetization.average()
        err = max_relative_error(result=av_mag,
                                 wanted=np.array([0.89947968, 0.23352228, -0.00010287]))
        assert err < RTOL
    
    def test_PBC_run_corners(self):
        """Perform the same comparison with the PBC as in mumax³ with the magnet in the corners."""
        world_PBC, magnet_PBC = simulate_PBC(True)
        world_PBC.timesolver.run(1e-9)
        av_mag = magnet_PBC.magnetization.average()
        err = max_relative_error(result=av_mag,
                                 wanted=np.array([0.89947968, 0.23352228, -0.00010287]))
        assert err < RTOL
