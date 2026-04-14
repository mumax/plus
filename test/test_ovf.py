from mumaxplus import *
import pyovf
import numpy as np

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
        self.aex = self.magnet.aex.eval()
        print()

        self.magnet.magnetization.save_ovf(self.name)
        self.magnet.magnetization.save_ovf()

        self.magnet.aex.save_ovf()

        self.ovf_m = pyovf.read(self.name)
        self.ovf_aex = pyovf.read(self.magnet.aex.name + ".ovf")

    def test_read(self):
        self.magnet.magnetization.load_ovf()
        mag = self.magnet.magnetization.eval()
        self.magnet.aex.load_ovf()
        aex = self.magnet.aex.eval()
        assert max_relative_error(mag, self.magnetization) < RTOL
        assert max_relative_error(aex, self.aex) < RTOL
    
    def test_read_name(self):
        self.magnet.magnetization.load_ovf(self.name)
        mag = self.magnet.magnetization.eval()
        assert max_relative_error(mag, self.magnetization) < RTOL

    def test_size(self):
        cx_ovf_m, cy_ovf_m, cz_ovf_m = self.ovf_m.xstepsize, self.ovf_m.ystepsize, self.ovf_m.zstepsize
        cx_ovf_aex, cy_ovf_aex, cz_ovf_aex = self.ovf_aex.xstepsize, self.ovf_aex.ystepsize, self.ovf_aex.zstepsize
        cx, cy, cz = self.world.cellsize
        assert max_relative_error(cx_ovf_m, cx) < RTOL and max_relative_error(cy_ovf_m, cy) < RTOL and max_relative_error(cz_ovf_m, cz) < RTOL
        assert max_relative_error(cx_ovf_aex, cx) < RTOL and max_relative_error(cy_ovf_aex, cy) < RTOL and max_relative_error(cz_ovf_aex, cz) < RTOL


    def test_grid(self):
        nx_ovf_m, ny_ovf_m, nz_ovf_m = self.ovf_m.xnodes, self.ovf_m.ynodes, self.ovf_m.znodes
        nx_ovf_aex, ny_ovf_aex, nz_ovf_aex = self.ovf_aex.xnodes, self.ovf_aex.ynodes, self.ovf_aex.znodes
        nx, ny, nz = self.magnet.grid.size
        assert nx_ovf_m == nx and ny_ovf_m == ny and nz_ovf_m == nz
        assert nx_ovf_aex == nx and ny_ovf_aex == ny and nz_ovf_aex == nz

    def test_title(self):
        assert self.ovf_m.Title == self.magnet.magnetization.name
        assert self.ovf_aex.Title == self.magnet.aex.name
