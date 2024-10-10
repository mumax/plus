import pytest
import numpy as np
import math

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
P = 2  # periods
A = 4  # amplitude
N = 128  # number of 1D cells

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


def create_magnet(d_comp, cos_comp):
    """Makes a world with a 1D magnet in the d_comp direction.
    """
    gridsize, gridsize_magnet, pbc_repetitions = [0, 0, 0], [1, 1, 1], [0, 0, 0]
    gridsize[d_comp], pbc_repetitions[d_comp] = N, 1  # set for PBC grid
    gridsize_magnet[d_comp] = N

    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    
    magnet =  Ferromagnet(world, Grid(gridsize_magnet))
    magnet.enable_elastodynamics = True

    magnet.enable_elastodynamics = True  # just in case

    L = N*cellsize[d_comp]
    k = P*2*math.pi/L

    def displacement_func(x, y, z):
        u = [0,0,0]
        comp = (x,y,z)[d_comp]
        u[d_comp] = A * math.sin(k * comp)
        u[cos_comp] = A * math.cos(k * comp)
        return tuple(u)

    magnet.elastic_displacement = displacement_func
    strain_num = magnet.strain_tensor.eval()
    
    strain_anal = np.zeros(shape=strain_num.shape)
    strain_anal[d_comp,...] = k * magnet.elastic_displacement.eval()[cos_comp,...]
    strain_anal[2 + d_comp + cos_comp,...] = - 0.5 * k * magnet.elastic_displacement.eval()[d_comp,...]
    assert max_semirelative_error(strain_num, strain_anal) < RTOL


def test_sinx_cosx_0():
    d_comp = 0
    create_magnet(d_comp, (d_comp+1)%3)

def test_0_siny_cosy():
    d_comp = 1
    create_magnet(d_comp, (d_comp+1)%3)

def test_cosz_0_sinz():
    d_comp = 2
    create_magnet(d_comp, (d_comp+1)%3)

def test_sinx_0_cosx():
    d_comp = 0
    create_magnet(d_comp, (d_comp+2)%3)

def test_cosy_siny_0():
    d_comp = 1
    create_magnet(d_comp, (d_comp+2)%3)

def test_0_cosz_sinz():
    d_comp = 2
    create_magnet(d_comp, (d_comp+2)%3)
