from mumaxplus import World, Ferromagnet, Grid
import pyovf
import numpy as np

RTOL = 2e-7

def max_relative_error(result, wanted):
    return np.max(np.abs(result - wanted) / np.abs(wanted))

class TestOVF:
    def setup_class(self):
        self.name = "test.ovf"
        cx, cy, cz = 4e-9, 2e-9, 5e-9
        nx, ny, nz = 32, 16, 8

        self.world = World(cellsize=(cx, cy, cz))

        self.magnet = Ferromagnet(self.world, Grid((nx, ny, nz)))

        self.magnet.enable_elastodynamics = True
        self.magnet.magnetization = (1,0,0)
        self.magnet.aex = 13e-12
        self.magnet.elastic_displacement = np.random.rand(3,nz,ny,nx)*1e-14

        self.world.timesolver.run(0.5e-9)

        self.magnetization = self.magnet.magnetization.eval()
        self.aex = self.magnet.aex.eval()
        self.strain = self.magnet.strain_tensor.eval()

        # Save 1D FieldQuantity
        self.magnet.aex.save_ovf()

        # Save 3D FieldQuantity
        self.magnet.magnetization.save_ovf(self.name)
        self.magnet.magnetization.save_ovf()
        
        # Save 6D FieldQuantity
        self.magnet.strain_tensor.save_ovf()

        self.ovf_m = pyovf.read(self.name)
        self.ovf_aex = pyovf.read(self.magnet.aex.name.replace(":", "_") + ".ovf")
        self.ovf_strain = pyovf.read(self.magnet.strain_tensor.name.replace(":", "_") + ".ovf")

    def test_1D(self):
        self.magnet.aex = 1e-12 # Change aex
        self.magnet.aex.load_ovf()
        aex = self.magnet.aex.eval()
        assert max_relative_error(aex, self.aex) < RTOL

    def test_3D(self):
        self.magnet.magnetization = (0,0,1) # Change magnetization
        self.magnet.magnetization.load_ovf()
        mag = self.magnet.magnetization.eval()
        assert max_relative_error(mag, self.magnetization) < RTOL

    def test_6D(self):
        nx, ny, nz = self.magnet.grid.size
        self.magnet.elastic_displacement = np.random.rand(3,nz,ny,nx)*1e-14 # Change strain_tensor
        strain = np.ascontiguousarray(np.moveaxis(self.ovf_strain.data, -1, 0))
        assert max_relative_error(strain, self.strain) < RTOL
    
    def test_read_name(self):
        self.magnet.magnetization = (0,0,1) # Change magnetization
        self.magnet.magnetization.load_ovf(self.name)
        mag = self.magnet.magnetization.eval()
        assert max_relative_error(mag, self.magnetization) < RTOL

    def test_size(self):
        cx_ovf_m, cy_ovf_m, cz_ovf_m = self.ovf_m.xstepsize, self.ovf_m.ystepsize, self.ovf_m.zstepsize
        cx_ovf_aex, cy_ovf_aex, cz_ovf_aex = self.ovf_aex.xstepsize, self.ovf_aex.ystepsize, self.ovf_aex.zstepsize
        cx_ovf_strain, cy_ovf_strain, cz_ovf_strain = self.ovf_strain.xstepsize, self.ovf_strain.ystepsize, self.ovf_strain.zstepsize
        cx, cy, cz = self.world.cellsize
        assert max_relative_error(cx_ovf_m, cx) < RTOL and max_relative_error(cy_ovf_m, cy) < RTOL and max_relative_error(cz_ovf_m, cz) < RTOL
        assert max_relative_error(cx_ovf_aex, cx) < RTOL and max_relative_error(cy_ovf_aex, cy) < RTOL and max_relative_error(cz_ovf_aex, cz) < RTOL
        assert max_relative_error(cx_ovf_strain, cx) < RTOL and max_relative_error(cy_ovf_strain, cy) < RTOL and max_relative_error(cz_ovf_strain, cz) < RTOL


    def test_grid(self):
        nx_ovf_m, ny_ovf_m, nz_ovf_m = self.ovf_m.xnodes, self.ovf_m.ynodes, self.ovf_m.znodes
        nx_ovf_aex, ny_ovf_aex, nz_ovf_aex = self.ovf_aex.xnodes, self.ovf_aex.ynodes, self.ovf_aex.znodes
        nx_ovf_strain, ny_ovf_strain, nz_ovf_strain = self.ovf_strain.xnodes, self.ovf_strain.ynodes, self.ovf_strain.znodes
        nx, ny, nz = self.magnet.grid.size
        assert nx_ovf_m == nx and ny_ovf_m == ny and nz_ovf_m == nz
        assert nx_ovf_aex == nx and ny_ovf_aex == ny and nz_ovf_aex == nz
        assert nx_ovf_strain == nx and ny_ovf_strain == ny and nz_ovf_strain == nz

    def test_title(self):
        assert self.ovf_m.Title == self.magnet.magnetization.name
        assert self.ovf_aex.Title == self.magnet.aex.name
        assert self.ovf_strain.Title == self.magnet.strain_tensor.name
