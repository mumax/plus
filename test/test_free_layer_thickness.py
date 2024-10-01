"""This test checks if the Slonczewski torque remains the same
if empty layers are added to the system.
"""


import pytest
import numpy as np

from mumaxplus import Ferromagnet, Grid, World


RTOL = 1e-5  # 0.001%

# Arbitrary parameters, resulting in a non-zero Slonczewski torque
magnetization = (np.random.uniform(), np.random.uniform(), np.random.uniform())
magnetization /= np.linalg.norm(magnetization)
msat = 4.3
jcur = (0,0,5.4)
pol = 0.6
fixed_layer = (1,0,1)
free_layer_thickness = 1e-9


def relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return relerr

def test_free_layer():
    # === Create a simulation ==
    world_wanted = World(cellsize=(1e-9, 1e-9, 1e-9))
    magnet_wanted = Ferromagnet(world_wanted, Grid((1, 1, 2)))

    magnet_wanted.msat = msat
    magnet_wanted.jcur = jcur
    magnet_wanted.pol = pol
    magnet_wanted.fixed_layer = fixed_layer
    magnet_wanted.free_layer_thickness = free_layer_thickness
    magnet_wanted.magnetization = magnetization

    torque_wanted = magnet_wanted.spin_transfer_torque.eval()[:,0,0,0]

    # === Create the same simulation with empty layers ===
    world_result = World(cellsize=(1e-9, 1e-9, 1e-9))
    magnet_result = Ferromagnet(world_result, Grid((1, 1, 4)), geometry=np.array([[[True]], [[True]], [[False]], [[False]]]))

    magnet_result.msat = msat
    magnet_result.jcur = jcur
    magnet_result.pol = pol
    magnet_result.fixed_layer = fixed_layer
    magnet_result.free_layer_thickness = free_layer_thickness
    magnet_result.magnetization = magnetization

    torque_result = magnet_wanted.spin_transfer_torque.eval()[:,0,0,0]

    err = relative_error(torque_result, torque_wanted)
    assert err < RTOL
