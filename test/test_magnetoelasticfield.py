import numpy as np
import math

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
P = 2  # periods
A = 4  # amplitude
N = 128  # number of 1D cells
msat = 800e3
B = -8.8e6

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


def create_magnet(d_comp, m_comp):
    """Makes a world with a 1D magnet in the d_comp 
    and a magnetization in the m_comp direction.
    """
    gridsize, gridsize_magnet, pbc_repetitions = [0, 0, 0], [1, 1, 1], [0, 0, 0]
    m = [0, 0, 0]
    m[m_comp] = 1
    gridsize[d_comp], pbc_repetitions[d_comp] = N, 1  # set for PBC grid
    gridsize_magnet[d_comp] = N

    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    
    magnet =  Ferromagnet(world, Grid(gridsize_magnet))
    magnet.enable_elastodynamics = True

    magnet.magnetization = m
    magnet.msat = msat

    return magnet


def sine_displacement(magnet, i_comp, j_comp, B1=0, B2=0, Bc=0):
    """Creates a displacement in the j_comp direction following a sine along the
    i_comp direction. Then calculates and compares analytical and numerical
    magnetoelastic field.
    """
    magnet.enable_elastodynamics = True  # just in case

    magnet.B1 = B1
    magnet.B2 = B2
    magnet.B_chiral = Bc

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
    index = [0, 0, 0]
    index[i_comp] = slice(None)  # index with length N
    
    cos[index[2], index[1], index[0]] = A * np.cos(k * cellsize[i_comp] * np.arange(0,N))

    if i_comp == j_comp:
        strain_anal[j_comp,...] = k * cos
    else:
        strain_anal[2 + i_comp + j_comp,...] = 0.5 * k * cos
    
    B_num = magnet.magnetoelastic_field.eval()
    B_anal = np.zeros(shape=B_num.shape)

    m = magnet.magnetization.eval()
    for i in range(3):
        ip1 = (i+1)%3
        ip2 = (i+2)%3

        B_anal[i,...] = - 2  / msat * (
            B1 *  strain_anal[i,...] * m[i,...] + 
            Bc * (-strain_anal[ip1,...] + strain_anal[ip2,...]) * m[i,...] +
            B2 * (strain_anal[i+ip1+2,...] * m[ip1,...] + 
                  strain_anal[i+ip2+2,...] * m[ip2,...]))

    assert max_semirelative_error(B_num, B_anal) < RTOL

# --- B1 ---

def test_mx_Exx_B1():
    m_comp, i, j = 0, 0, 0
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B1=B)

def test_my_Eyy_B1():
    m_comp, i, j = 1, 1, 1
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B1=B)

def test_mz_Ezz_B1():
    m_comp, i, j = 2, 2, 2
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B1=B)

# --- B2 ---

def test_my_Exy_B2():
    m_comp, i, j = 1, 0, 1
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B2=B)

def test_mz_Exz_B2():
    m_comp, i, j = 2, 0, 2
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B2=B)

def test_mx_Exy_B2():
    m_comp, i, j = 0, 0, 1
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B2=B)

def test_mz_Eyz_B2():
    m_comp, i, j = 2, 1, 2
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B2=B)

def test_mx_Exz_B2():
    m_comp, i, j = 0, 0, 2
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B2=B)

def test_my_Eyz_B2():
    m_comp, i, j = 1, 1, 2
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, B2=B)

# --- Bc ---

def test_mx_Eyy_Bc():
    m_comp, i, j = 0, 1, 1
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, Bc=B)

def test_mx_Ezz_Bc():
    m_comp, i, j = 0, 2, 2
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, Bc=B)

def test_my_Exx_Bc():
    m_comp, i, j = 1, 0, 0
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, Bc=B)

def test_my_Ezz_Bc():
    m_comp, i, j = 1, 2, 2
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, Bc=B)

def test_mz_Exx_Bc():
    m_comp, i, j = 2, 0, 0
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, Bc=B)

def test_mz_Eyy_Bc():
    m_comp, i, j = 2, 1, 1
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, Bc=B)


def test_random():
    """Test with a random magnetization and displacement.
    Trust the numerical strain.
    """
    gridsize, gridsize_magnet, pbc_repetitions = [0, 0, 0], [1, 1, 1], [0, 0, 0]

    gridsize[0], pbc_repetitions[0] = N, 1  # set for PBC grid
    gridsize_magnet[0] = N

    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    magnet =  Ferromagnet(world, Grid(gridsize_magnet))
    magnet.enable_elastodynamics = True

    magnet.msat = msat
    B1, B2, Bc = B, 0.5*B, 0.3*B
    magnet.B1 = B1
    magnet.B2 = B2
    magnet.B_chiral = Bc

    L = N*cellsize[0]
    k = P*2*math.pi/L

    def displacement_func(x, y, z):
        return tuple(np.random.rand(3))

    magnet.elastic_displacement = displacement_func
    strain = magnet.strain_tensor.eval()
    
    B_num = magnet.magnetoelastic_field.eval()
    B_anal = np.zeros(shape=B_num.shape)

    m = magnet.magnetization.eval()
    for i in range(3):
        ip1 = (i+1)%3
        ip2 = (i+2)%3

        B_anal[i,...] = - 2  / msat * (
            B1 *  strain[i,...] * m[i,...] + 
            Bc * (-strain[ip1,...] + strain[ip2,...]) * m[i,...] +
            B2 * (strain[i+ip1+2,...] * m[ip1,...] + 
                  strain[i+ip2+2,...] * m[ip2,...]))

    assert max_semirelative_error(B_num, B_anal) < RTOL