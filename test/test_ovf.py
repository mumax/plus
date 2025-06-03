"""This file tests the reading and writing of OVF 2.0 text files
"""

import numpy as np
import os

from mumaxplus import Grid, World, Ferromagnet

ATOL = 2e-7

def max_absolute_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    return np.max(err)

class TestOvf:
    def setup_class(self):
        self.world = World((1e-9, 3e-9, 4e-9))
        self.magnet = Ferromagnet(self.world, Grid((3, 2, 1)))
        self.magnet.magnetization = (1, 0.1, 0)
        self.magnet.minimize()
        
        self.magnet.ku1 = 2
        self.magnet.anisU = (1,0,0)

        print(self.magnet.magnetization.eval())

        self.magnet.magnetization.write_ovf("test_m.ovf")
        self.magnet.ku1.write_ovf("test_ku1.ovf")
        self.magnet.anisU.write_ovf("test_anisU.ovf")
    
    def test_read_m(self):
        print("helloooo")
        world = World((1e-9, 3e-9, 4e-9))
        magnet = Ferromagnet(world, Grid((3, 2, 1)))
        print("hello")
        magnet.magnetization.read_ovf("test_m.ovf")
        os.remove("test_m.ovf")
        
        assert max_absolute_error(magnet.magnetization.eval(), self.magnet.magnetization.eval()) < ATOL
    
    def test_read_ku1(self):
        world = World((1e-9, 3e-9, 4e-9))
        magnet = Ferromagnet(world, Grid((3, 2, 1)))
        magnet.ku1.read_ovf("test_ku1.ovf")
        os.remove("test_ku1.ovf")
        
        assert max_absolute_error(magnet.ku1.eval(), self.magnet.ku1.eval()) < ATOL
    
    def test_read_anisU(self):
        world = World((1e-9, 3e-9, 4e-9))
        magnet = Ferromagnet(world, Grid((3, 2, 1)))
        magnet.anisU.read_ovf("test_anisU.ovf")
        os.remove("test_anisU.ovf")
        
        assert max_absolute_error(magnet.anisU.eval(), self.magnet.anisU.eval()) < ATOL