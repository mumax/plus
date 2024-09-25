import pytest
import numpy as np

from mumaxplus import Ferromagnet, Grid, World

"""This test checks if the Slonczewski torque remains the same
if empty layers are added to the system.
"""

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

class TestFreeLayer:
    def setup_class(self):
        # === Create a simulation ==
        world_wanted = World(cellsize=(1e-9, 1e-9, 1e-9))
        magnet_wanted = Ferromagnet(world_wanted, Grid((1, 1, 2)))

        magnet_wanted.msat = msat
        magnet_wanted.jcur = jcur
        magnet_wanted.pol = pol
        magnet_wanted.fixed_layer = fixed_layer
        magnet_wanted.free_layer_thickness = free_layer_thickness
        magnet_wanted.magnetization = magnetization

        self.torque_wanted = np.array([magnet_wanted.spin_transfer_torque.eval()[0,0,0,0],
                                  magnet_wanted.spin_transfer_torque.eval()[1,0,0,0],
                                  magnet_wanted.spin_transfer_torque.eval()[2,0,0,0]])
        
        # === Create the same simulation with empty layers ===
        world_result = World(cellsize=(1e-9, 1e-9, 1e-9))
        magnet_result = Ferromagnet(world_result, Grid((1, 1, 4)), geometry=np.array([[[True]], [[True]], [[False]], [[False]]]))

        magnet_result.msat = msat
        magnet_result.jcur = jcur
        magnet_result.pol = pol
        magnet_result.fixed_layer = fixed_layer
        magnet_result.free_layer_thickness = free_layer_thickness
        magnet_result.magnetization = magnetization

        self.torque_result = np.array([magnet_result.spin_transfer_torque.eval()[0,0,0,0],
                                  magnet_result.spin_transfer_torque.eval()[1,0,0,0],
                                  magnet_result.spin_transfer_torque.eval()[2,0,0,0]])
    
    def test_free_layer(self):
        err = relative_error(self.torque_result, self.torque_wanted)
        assert err < RTOL