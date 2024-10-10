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


def create_magnet(i_comp, j_comp):
    """Makes a world with a 1D magnet in the d_comp direction.
    """
    gridsize, gridsize_magnet, pbc_repetitions = [0, 0, 0], [1, 1, 1], [0, 0, 0]
    gridsize[i_comp], pbc_repetitions[i_comp] = N, 1  # set for PBC grid
    gridsize_magnet[i_comp] = N

    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    
    magnet =  Ferromagnet(world, Grid(gridsize_magnet))
    magnet.enable_elastodynamics = True

    magnet.enable_elastodynamics = True  # just in case

    L = N*cellsize[i_comp]
    k = P*2*math.pi/L

    def displacement_func(x, y, z):
        u = [0,0,0]
        comp = (x,y,z)[i_comp]
        u[j_comp] = A * math.sin(k * comp)
        return tuple(u)

    magnet.elastic_displacement = displacement_func
    strain_num = magnet.strain_tensor.eval()
    
    strain_anal = np.zeros(shape=strain_num.shape)

    # derivative of sine is cosine
    cos = np.zeros(shape=strain_anal[0,...].shape)
    index = [slice(None), slice(None), slice(None)]
    index[(i_comp+1)%3] = 0  # indexes with length 1
    index[(i_comp+2)%3] = 0  # indexes with length 1
    cos[index[2], index[1], index[0]] = A * np.cos(k * cellsize[i_comp] * np.arange(0,N))
    if i_comp == j_comp:
        strain_anal[j_comp,...] = k * cos
    else:
        strain_anal[2 + i_comp + j_comp,...] = 0.5 * k * cos
    
    assert max_semirelative_error(strain_num, strain_anal) < RTOL


def test_Exx():
    i, j = 0, 0
    create_magnet(i, j)

def test_Eyy():
    i, j = 1, 1
    create_magnet(i, j)

def test_Ezz():
    i, j = 2, 2
    create_magnet(i, j)

def test_Exy():
    i, j = 0, 1
    create_magnet(i, j)

def test_Exz():
    i, j = 0, 2
    create_magnet(i, j)

def test_Eyz():
    i, j = 1, 2
    create_magnet(i, j)
