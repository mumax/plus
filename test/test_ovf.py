from mumaxplus import *
import pyovf
import numpy as np
from mumaxplus.util import plot_field

RTOL = 2e-7

def max_relative_error(result, wanted):
    return np.max(np.abs(result - wanted) / np.abs(wanted))

class TestOVF:
    def setup_class(self):
        self.name = "test.ovf"
        cx, cy, cz = 4e-9, 2e-9, 5e-9
        nx, ny, nz = 128, 32, 1

        self.world = World(cellsize=(cx, cy, cz))

        self.magnet = Ferromagnet(self.world, Grid((nx, ny, nz)))

        self.magnet.magnetization = (1,0,0)
        self.magnet.msat = 800e3
        self.magnet.aex = 13e-12
        self.magnet.alpha = 0.02

        self.world.timesolver.run(0.5e-9)

        self.magnetization = self.magnet.magnetization.eval()

        self.magnet.magnetization.save_ovf(self.name)
        self.magnet.magnetization.save_ovf()

        self.ovf = pyovf.read(self.name)

    def test_read(self):
        self.magnet.magnetization.load_ovf()
        mag = self.magnet.magnetization.eval()
        assert max_relative_error(mag, self.magnetization) < RTOL
    
    def test_read_name(self):
        self.magnet.magnetization.load_ovf(self.name)
        mag = self.magnet.magnetization.eval()
        assert max_relative_error(mag, self.magnetization) < RTOL

    def test_size(self):
        cx_ovf, cy_ovf, cz_ovf = self.ovf.xstepsize, self.ovf.ystepsize, self.ovf.zstepsize
        cx, cy, cz = self.world.cellsize
        assert max_relative_error(cx_ovf, cx) < RTOL and max_relative_error(cy_ovf, cy) < RTOL and max_relative_error(cz_ovf, cz) < RTOL

    def test_grid(self):
        nx_ovf, ny_ovf, nz_ovf = self.ovf.xnodes, self.ovf.ynodes, self.ovf.znodes
        nx, ny, nz = self.magnet.grid.size
        assert nx_ovf == nx and ny_ovf == ny and nz_ovf == nz

    def test_title(self):
        assert self.ovf.Title == self.magnet.magnetization.name
