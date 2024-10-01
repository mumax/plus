import numpy as np

from mumaxplus import Ferromagnet, Grid, World, _cpp
from mumaxplus.util.constants import MU0
from mumaxplus.util.shape import Ellipse


def demag_field_py(magnet):
    kernel = _cpp._demag_kernel(magnet._impl)
    mag = magnet.msat.average() * magnet.magnetization.get()
    # add padding to the magnetization so that the size of magnetization
    # matches the size of the kernel
    pad = (
        (0, kernel.shape[1] - mag.shape[1]),
        (0, kernel.shape[2] - mag.shape[2]),
        (0, kernel.shape[3] - mag.shape[3]),
    )
    m = np.pad(mag, ((0, 0), *pad), "constant")

    # fourier transform of the magnetization and the kernel
    m = np.fft.fftn(m, axes=(1, 2, 3))
    kxx, kyy, kzz, kxy, kxz, kyz = np.fft.fftn(kernel, axes=(1, 2, 3))

    # apply the kernel and perform inverse fft
    hx = np.fft.ifftn(m[0] * kxx + m[1] * kxy + m[2] * kxz)
    hy = np.fft.ifftn(m[0] * kxy + m[1] * kyy + m[2] * kyz)
    hz = np.fft.ifftn(m[0] * kxz + m[1] * kyz + m[2] * kzz)

    # return the real part
    mu0 = 4 * np.pi * 1e-7
    h = -mu0 * np.array([hx, hy, hz]).real
    return h[
        :,
        (h.shape[1] - mag.shape[1]) :,
        (h.shape[2] - mag.shape[2]) :,
        (h.shape[3] - mag.shape[3]) :,
    ]


class TestDemag:
    def test_demagfield(self):
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((16, 4, 3)))
        wanted = demag_field_py(magnet)
        result = magnet.demag_field.eval()
        err = np.max(np.abs((wanted - result) / result))
        assert err < 2e-3

class TestDemag2D:
    """Test demag field for thin film using flat cells.
	Kernel should be approximately 0, 0, -1 
    """
    def setup_class(self):
        cellsize = (1e-9, 1e-9, 0.5e-9)
        gridsize = (128, 128, 1)

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-2  # not perfectly 0, finite film
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-2  # not perfectly 0, finite film
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1) < 2e-2  # not perfectly -1, finite film


class TestDemagOdd:
    """Test demag tensor of cube for odd N."""
    def setup_class(self):
        N, c = 5, 1e-9
        self.ATOL = 5e-4

        cellsize = (c, c, c)
        gridsize = (N, N, N)

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0] - -1./3.) < self.ATOL
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1] - -1./3.) < self.ATOL
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1./3.) < self.ATOL


class TestDemagRod:
    """Test demag field for long rod
    Kernel should be approximately -.5 -.5 0
    """
    def setup_class(self):
        self.ATOL = 1e-2

        cellsize = (1e-9, 1e-9, 2e-9)
        gridsize = (2, 2, 64)

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0] - -0.496) < self.ATOL
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1] - -0.496) < self.ATOL
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL


class TestDemagCube:
    """Test demag field of cube
	demag tensor should be -1/3, -1/3, -1/3
    """
    def setup_class(self):
        N, c = 2, 1e-9
        self.ATOL = 5e-4

        cellsize = (c, c, c)
        gridsize = (N, N, N)

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0] - -1./3.) < self.ATOL
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1] - -1./3.) < self.ATOL
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1./3.) < self.ATOL


class TestDemag2DPBC:
    """Test demag field for thin film using PBC to extend the film along Y.
	Kernel should be approximately 0, 0, -1 
    """
    def setup_class(self):
        cellsize = (1e-9, 1e-9, 0.5e-9)
        gridsize = (128, 2, 1)

        self.world = World(cellsize, mastergrid=Grid((0,2,0)), pbc_repetitions=(0,32,0))
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-2  # Not perfectly 0, finite film
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-2  # Not perfectly 0, finite film
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1) < 1e-2  # Not perfectly -1, finite film


# FAIL
class TestDemagSmall:
    """Regression test for demag of a small 2D magnet.
	This is sensitive to minor range mistakes in the convolution which
	may not show up clearly for large geometries.
	Values from mumax3, we test that they don't silently change.
    """
    def setup_class(self):
        self.ATOL = 1e-6

        cellsize = (1e-9, 2e-9, 0.5e-9)
        gridsize = (3, 4, 1)

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0] - -0.15768295526) < self.ATOL
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1] - 0.05676037073) < self.ATOL
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -0.78447151184) < self.ATOL


class TestDemag2DPBC2Even:
    """Test demag field for thin film. 
	Kernel should be approximately 0, 0, -1
    """
    def setup_class(self):
        cellsize = (1e-9, 1e-9, 0.5e-9)
        gridsize = (2, 128, 1)

        self.world = World(cellsize, mastergrid=Grid((2,0,0)), pbc_repetitions=(32,0,0))
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-2  # Not perfectly 0, finite film
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-2  # Not perfectly 0, finite film
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1) < 1e-2  # Not perfectly -1, finite film


# FAIL
class TestDemag2DPBC2Odd:
    """Test demag field for thin film. 
	Kernel should be approximately 0, 0, -1
    Repeat TestDemag2DPBC2Even with an odd number of cells along x.
    This led to a fft size mismatch panic in mumax3.10 and before.
    """
    def setup_class(self):
        cellsize = (1e-9, 1e-9, 0.5e-9)
        gridsize = (3, 128, 1)

        self.world = World(cellsize, mastergrid=Grid((3,0,0)), pbc_repetitions=(32,0,0))
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-2  # Not perfectly 0, finite film
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-2  # Not perfectly 0, finite film
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1) < 1e-2  # Not perfectly -1, finite film


class TestDemag3DFilm:
    """Test demag field for thin film with 3D discretization.
	Kernel should be approximately 0, 0, -1 
    """
    def setup_class(self):
        cellsize = (1e-9, 1.2e-9, 0.5e-9)
        gridsize = (64, 48, 2)

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0]) < 3e-2  # Not perfectly 0, finite film
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1]) < 3e-2  # Not perfectly 0, finite film
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1) < 6e-2  # Not perfectly -1, finite film


class TestDemagRegion:
    """Test demag field with regions
    """
    def setup_class(self):
        cellsize = (1e-9, 1e-9, 0.5e-9)
        gridsize = (64, 64, 2)

        self.world = World(cellsize)

        B = 2/MU0
        r = Ellipse(10e-9, 20e-9)
        def region(x,y,z):
            if r(x,y,z):
                return 1
            elif x <= 0:
                return 2
            else:
                return 3
        
        self.magnet = Ferromagnet(self.world, Grid(gridsize), regions=region)
        self.magnet.msat.set_in_region(1, B)
        self.magnet.msat.set_in_region(2, B)
        self.magnet.msat.set_in_region(3, B)

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0]) < 0.1  # Not perfectly 0, finite film
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-8
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-8
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-8
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1]) < 0.1 # Not perfectly 0, finite film
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-8

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-8
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-8
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -2) < 0.1  # Not perfectly -1, finite film
    
    def test_zz2(self):
        self.magnet.msat = 1/MU0
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1) < 0.1  # Not perfectly -1, finite film


class TestDemagRodPBC:
    """Test demag field for rod, using PBC to make the rod long.
	Kernel should be approximately -.5 -.5 0
    """
    def setup_class(self):
        self.ATOL = 1e-2

        cellsize = (1e-9, 1e-9, 2e-9)
        gridsize = (2,2,2)

        self.world = World(cellsize, mastergrid=Grid((0,0,2)), pbc_repetitions=(0,0,16))
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0] - -0.496) < self.ATOL
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1] - -0.496) < self.ATOL
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL


# FAIL
class TestDemag2DLong:
    """Test demag field for thin film with elongated size and cells.
	Kernel should be approximately 0, 0, -1
    """
    def setup_class(self):
        cellsize = (0.5e-9, 2e-9, 1e-9)
        gridsize = (1024,125,2)

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-2  # Not perfectly 0, finite film
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-2  # Not perfectly 0, finite film
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < 1e-9

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < 1e-9
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < 1e-9
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -1) < 2e-2  # Not perfectly 0, finite film


# FAIL
class TestDemagSmall3D:
    """Regression test for demag of a small 3D magnet.
	This is sensitive to minor range mistakes in the convolution which
	may not show up clearly for large geometries.
	Values from mumax3, we test that they don't silently change.
    """
    def setup_class(self):
        self.ATOL = 1e-6

        cellsize = (1e-9, 2e-9, 3e-9)
        gridsize = (3,4,2)

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))

        self.magnet.msat = 1/MU0

    def test_xx(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[0] - -0.52710835138) < self.ATOL
    
    def test_xy(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_xz(self):
        self.magnet.magnetization = (1,0,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL
    
    def test_yx(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_yy(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[1] - -0.20121671756) < self.ATOL
    
    def test_yz(self):
        self.magnet.magnetization = (0,1,0)
        assert abs(self.magnet.demag_field.average()[2]) < self.ATOL

    def test_zx(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[0]) < self.ATOL
    
    def test_zy(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[1]) < self.ATOL
    
    def test_zz(self):
        self.magnet.magnetization = (0,0,1)
        assert abs(self.magnet.demag_field.average()[2] - -0.27112555503) < self.ATOL
