import numpy as np
import math

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
P = 2  # sine periods
N = 128  # number of 1D cells
B = -8.8e6

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


def create_magnet(d_comp, B1, B2):
    """Makes a world with a 1D magnet in the d_comp direction.
    """
    gridsize, gridsize_magnet, pbc_repetitions = [0, 0, 0], [1, 1, 1], [0, 0, 0]
    gridsize[d_comp], pbc_repetitions[d_comp] = N,1  # set for PBC grid
    gridsize_magnet[d_comp] = N

    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    
    magnet =  Ferromagnet(world, Grid(gridsize_magnet))
    magnet.enable_elastodynamics = True

    magnet.B1 = B1
    magnet.B2 = B2

    return world, magnet


def sine_force(magnet, d_comp, comp_cos, B1, B2):
    """Creates the magnetization following a sine in the d_comp direction
    and cosine in another direction it then calculates the magnetoelasticforce.
    """
    magnet.enable_elastodynamics = True  # just in case

    L = N*cellsize[d_comp]
    k = P*2*math.pi/L

    def magnetization_func(x, y, z):
        m = [0,0,0]
        comp = (x,y,z)[d_comp]
        m[d_comp] = math.sin(k * comp)
        m[comp_cos] = math.cos(k * comp)
        return tuple(m)

    magnet.magnetization = magnetization_func

    force_num = magnet.magnetoelastic_force.eval()
    
    force_anal = np.zeros(shape=force_num.shape)
    force_anal[d_comp,...] = 2 * B1 * k * magnet.magnetization.eval()[d_comp,...] * magnet.magnetization.eval()[comp_cos,...]
    force_anal[comp_cos,...] = B2 * k * (magnet.magnetization.eval()[comp_cos,...]**2 - magnet.magnetization.eval()[d_comp,...]**2)
    
    assert max_semirelative_error(force_num, force_anal) < RTOL


def test_sinx_cosx_0_B1():
    d_comp = 0
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_0_siny_cosy_B1():
    d_comp = 1
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_cosz_0_sinz_B1():
    d_comp = 2
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_sinx_cosx_0_B2():
    d_comp = 0
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_0_siny_cosy_B2():
    d_comp = 1
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_cosz_0_sinz_B2():
    d_comp = 2
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_sinx_0_cosx_B2():
    d_comp = 0
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_cosy_siny_0_B2():
    d_comp = 1
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_0_cosz_sinz_B2():
    d_comp = 2
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, B1, B2)
    sine_force(magnet, d_comp, (d_comp+2)%3, B1, B2)
