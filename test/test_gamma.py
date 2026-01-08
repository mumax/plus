from mumaxplus import Grid, World, Ferromagnet
import numpy as np

def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

def test_gyromagnetic_ratio_LLG_torque():
    """Reduce the gyromagnetic ratio in a copy of a magnet and compare the torques."""

    world1 = World((1e-9, 1e-9, 1e-9))
    world2 = World((1e-9, 1e-9, 1e-9))
    magnet1 = Ferromagnet(world1, Grid((16, 32, 8)))
    magnet2 = Ferromagnet(world2, Grid((16, 32, 8)))

    for magnet in [magnet1, magnet2]:
        magnet.msat = 800e3
        magnet.aex = 13e-12
    
    magnet2.magnetization = magnet1.magnetization()
    magnet2.gamma = magnet1.gamma() * 2

    t1 = magnet1.torque()
    t2 = magnet2.torque()
    
    err = max_relative_error(t2 / 2, t1)
    assert err < 5e-4

def test_gyromagnetic_ratio_damping_torque():
    """Reduce the gyromagnetic ratio in a copy of a magnet and compare the torques."""

    world1 = World((1e-9, 1e-9, 1e-9))
    world2 = World((1e-9, 1e-9, 1e-9))
    magnet1 = Ferromagnet(world1, Grid((16, 32, 8)))
    magnet2 = Ferromagnet(world2, Grid((16, 32, 8)))

    for magnet in [magnet1, magnet2]:
        magnet.msat = 800e3
        magnet.aex = 13e-12
        magnet.alpha = 0.01
    
    magnet2.magnetization = magnet1.magnetization()
    magnet2.gamma = magnet1.gamma() * 2

    t1 = magnet1.damping_torque()
    t2 = magnet2.damping_torque()
    
    err = max_relative_error(t2 / 2, t1)
    assert err < 5e-4

def test_gyromagnetic_ratio_Slonczewski():
    """Slonczewski STT scales with the gyromagnetic ratio"""

    world = World((1e-9, 1e-9, 1e-9))
    magnet = Ferromagnet(world, Grid((10, 10, 1)))

    magnet.msat = 800e3
    magnet.alpha = 0.01

    magnet.jcur = (1, 1, 1)
    magnet.fixed_layer = (1, 1, 1)
    magnet.pol = 0.3
    magnet.Lambda = 5

    t1 = magnet.spin_transfer_torque()
    magnet.gamma = magnet.gamma() * 2
    t2 = magnet.spin_transfer_torque()

    err = max_relative_error(t2 / 2, t1)
    assert err < 1e-7