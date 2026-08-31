import numpy as np
import pytest
from mumaxplus import Antiferromagnet, Ferromagnet, Grid, World

cs = 1e-9
runtime = 1e-12

def make_ferromagnet(nx=64, ny=1, nz=1):
    world = World((cs, cs, cs))
    grid = Grid((nx, ny, nz))
    magnet = Ferromagnet(world, grid)
    return world, magnet

def set_parameters(magnet, comp):
    magnet.enable_demag = False
    magnet.enable_openbc = True
    magnet.msat  = 800e3
    magnet.aex   = 13e-12
    magnet.alpha = 0.1
    magnet.ku1   = 5e5
    anisU = [0.0, 0.0, 0.0]
    anisU[comp] = 1.0
    magnet.anisU = tuple(anisU)
    magnet.xi   = 0.2
    magnet.pol = 1


def dw_profile(magnet, comp, axis, width=5, center=None, minimize=True):
    nz, ny, nx = magnet.grid.shape

    n = (nx, ny, nz)[axis]
    if center is None:
        center = int(0.5 * n)

    def slc(start, stop):
        s = [slice(None), slice(None), slice(None)]
        s[2 - axis] = slice(start, stop)
        return tuple(s)

    m = np.zeros((3, nz, ny, nx))

    m[comp][slc(None, center - width)] = 1  # 'left' domain
    m[comp][slc(center + width, None)] = -1 # 'right' domain
    m[(comp + 1) % 3][slc(center - width, center + width)] = 1 # domain wall

    magnet.magnetization = m
    if minimize:
        magnet.minimize()

class TestValidArguments:
    def test_valid_boundary(self):
        world, _ = make_ferromagnet()
        for b in (0, 1):
            world.window.insert_magnetization(b, (1, 0, 0))  # no raise

    def test_invalid_boundary(self):
        world, _ = make_ferromagnet()
        for b in (-1, 4, 1.5):
            with pytest.raises((ValueError)):
                world.window.insert_magnetization(b, (1, 0, 0))

    def test_invalid_comp(self):
        world, _ = make_ferromagnet()
        for c in (-1, 4, 1.5):
            with pytest.raises((ValueError)):
                world.center_domain_wall(c)
class TestInitialConditions:
    def test_geometry(self):
        w, g = World((1, 1, 1)), Grid((10, 10, 10))
        geo = np.ones(g.shape)
        geo[0, 0, 0] = 0
        magnet = Ferromagnet(w, g, geometry=geo)
        with pytest.raises((RuntimeError)):
            w.center_domain_wall(0, 0)

    def test_regions(self):
        w, g = World((1, 1, 1)), Grid((10, 10, 10))
        reg = np.zeros(g.shape)
        reg[0, 0, 0] = 1
        magnet = Ferromagnet(w, g, regions=reg)
        with pytest.raises((RuntimeError)):
            w.center_domain_wall(0, 0)

    def test_multiple_magnets(self):
        world = World((1, 1, 1))
        magnet_1 = Ferromagnet(world, Grid((10, 10, 1)))
        magnet_2 = Antiferromagnet(world, Grid((10, 10, 1), origin=(0, 0, 1)))
        with pytest.raises((RuntimeError)):
            world.center_domain_wall(0, 0)

    def test_origin(self):
        origin = (5, 6, 7)
        cellsize = (2e-9, 3e-9, 4e-9)
        origin_phys = np.array(origin) * np.array(cellsize)
        world = World(cellsize)
        magnet = Ferromagnet(world, Grid((11, 12, 13), origin=origin))
        assert np.all(np.isclose(world.window.position, origin_phys, atol=cs/10))

    def test_offset_domain_wall(self):
        world, magnet = make_ferromagnet(nx=100)
        dw_profile(magnet, 0, 0, center=20)
        with pytest.raises((RuntimeError)):
            world.center_domain_wall(0, 0)

    def test_zero_average(self):
        Nx = 128
        world, magnet = make_ferromagnet(Nx, 1, 1)
        set_parameters(magnet, 0)
        dw_profile(magnet, 0, 0)

        world.center_domain_wall(0, 0)
        magnet.jcur = (1e12, 0, 0)
        world.timesolver.run(runtime)
        av = magnet.magnetization.average()[0]
        assert np.abs(av) < 4 / Nx

class TestShift:
    def setup_and_shift(self, comp=0, axis=0, nx=64, ny=1, nz=1, current=1e14):
        world, magnet = make_ferromagnet(nx, ny, nz)
        set_parameters(magnet, comp)
        dw_profile(magnet, comp, axis)

        jcur = [0, 0, 0]
        jcur[axis] = current
        magnet.jcur = jcur

        world.center_domain_wall(comp, axis)
        world.timesolver.run(runtime)
        return world.window.position
    
    def test_axis0(self):
        shift = self.setup_and_shift(comp=0, axis=0, nx=64, ny=1, nz=1)
        assert abs(shift[0]) >= cs
        assert shift[1] == 0.0
        assert shift[2] == 0.0

    def test_axis1(self):
        shift = self.setup_and_shift(comp=0, axis=1, nx=1, ny=64, nz=1)
        assert abs(shift[1]) >= cs
        assert shift[0] == 0.0
        assert shift[2] == 0.0

    def test_axis2(self):
        shift = self.setup_and_shift(comp=0, axis=2, nx=1, ny=1, nz=64)
        assert abs(shift[2]) >= cs
        assert shift[0] == 0.0
        assert shift[1] == 0.0

    def test_comp0(self):
        shift = self.setup_and_shift(comp=0, axis=0)
        assert abs(shift[0]) >= cs

    def test_comp1(self):
        shift = self.setup_and_shift(comp=1, axis=0)
        assert abs(shift[0]) >= cs

    def test_comp2(self):
        shift = self.setup_and_shift(comp=2, axis=0)
        assert abs(shift[0]) >= cs

    def test_no_current(self):
        shift = self.setup_and_shift(comp=0, axis=0, current=0)
        assert np.all(np.isclose(shift, 0.0, atol=cs/10))

    def test_opposite_current(self):
        shift_right = self.setup_and_shift(axis=0, current=1e14)
        shift_left  = self.setup_and_shift(axis=0, current=-1e14)
        assert np.sign(shift_right[0]) != np.sign(shift_left[0])

    def test_disable_motion(self):
        Nx = 64
        world, magnet = make_ferromagnet(Nx, 1, 1)
        set_parameters(magnet, 0)
        dw_profile(magnet, 0, 0)
        magnet.jcur = (1e14, 0, 0)

        world.center_domain_wall(0, 0)
        world.timesolver.run(runtime)
        assert magnet.magnetization.average()[0] < 4 / Nx

        world.window.disable_motion()

        world.timesolver.run(runtime)
        av = magnet.magnetization.average()[0]
        assert np.abs(av) > 4 / Nx

class TestInsertion:
    def setup(self, comp, axis=0):
        nx, ny, nz = 64, 1, 1
        world, magnet = make_ferromagnet(nx, ny, nz)
        set_parameters(magnet, comp)
        dw_profile(magnet, comp, axis)
        return world, magnet

    def test_carry_magnetization(self):
        comp, axis = 2, 0
        world, magnet = self.setup(comp, axis)
        left_init  = magnet.magnetization()[comp, :, :,  0].squeeze()
        right_init = magnet.magnetization()[comp, :, :, -1].squeeze()
        
        world.center_domain_wall(comp, axis)
        
        magnet.jcur = (1e12, 0, 0)
        world.timesolver.run(runtime)

        left  = magnet.magnetization()[comp, :, :,  0].squeeze()
        right = magnet.magnetization()[comp, :, :, -1].squeeze()

        assert np.allclose((left_init, right_init), (left, right), atol=cs/10)

    def test_insertion_values(self):
        comp, axis = 2, 0
        world, magnet = self.setup(comp, axis)
        left_init  = magnet.magnetization()[comp, :, :,  0].squeeze()
        right_init = magnet.magnetization()[comp, :, :, -1].squeeze()

        world.window.insert_magnetization(0, np.array([0, 0, left_init]))
        world.window.insert_magnetization(1, np.array([0, 0, right_init]))
        
        world.center_domain_wall(comp, axis)
        magnet.jcur = (10e12, 0, 0)
        world.timesolver.run(runtime)

        left  = magnet.magnetization()[comp, :, :,  0].squeeze()
        right = magnet.magnetization()[comp, :, :, -1].squeeze()
        assert np.allclose((left_init, right_init), (left, right), atol=cs/10)

class TestAntiferromagnet:

    def setup(self, comp=0, axis=0):
        nx, ny, nz = 64, 1, 1
        world = World((cs, cs, cs))
        magnet = Antiferromagnet(world, Grid((nx, ny, nz)))
        set_parameters(magnet, comp)

        magnet.afmex_cell = -10e-12
        magnet.afmex_nn = -5e-12

        dw_profile(magnet.sub1, comp, axis, minimize=False)
        magnet.sub2.magnetization = - magnet.sub1.magnetization()
        magnet.minimize()

        jcur = [0.0, 0.0, 0.0]
        jcur[axis] = 5e12
        magnet.sub1.jcur = tuple(jcur)
        magnet.sub2.jcur = tuple(jcur)

        return world, magnet

    def test_same_shift(self):
        comp, axis = 0, 0
        world, magnet = self.setup(comp, axis)
        m1_before = magnet.sub1.magnetization()
        m2_before = magnet.sub2.magnetization()

        world.center_domain_wall(comp, axis)
        world.timesolver.run(runtime)

        diff1 = np.abs(magnet.sub1.magnetization() - m1_before)
        diff2 = np.abs(magnet.sub2.magnetization() - m2_before)

        assert np.allclose(diff1, diff2, atol=cs/10)