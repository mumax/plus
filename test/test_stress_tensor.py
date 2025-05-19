import numpy as np

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
nx, ny, nz = 128, 64, 1  # number of 1D cells
msat = 800e3
C11, C12, C44 = 283e9, 58e9, 166e9

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

def test_stress():
    """Test if the stress is correctly calculated, given a random displacement.
    """
    world = World(cellsize)
    
    magnet =  Ferromagnet(world, Grid((nx, ny, nz)))
    magnet.enable_elastodynamics = True

    magnet.msat = msat

    magnet.C11 = C11
    magnet.C12 = C12
    magnet.C44 = C44

    def displacement_func(x, y, z):
        return tuple(np.random.rand(3))

    magnet.elastic_displacement = displacement_func
    strain = magnet.strain_tensor.eval()
        
    stress_num = magnet.stress_tensor.eval()
    stress_anal = np.zeros(shape=stress_num.shape)

    for i in range(3):
        ip1 = (i+1)%3
        ip2 = (i+2)%3

        stress_anal[i,...] = C11 * strain[i,...] + C12 * strain[ip1,...] + C12 * strain[ip2,...]
        stress_anal[i+3,...] = 2 * C44 * strain[i+3,...]  # using real strain

    assert max_semirelative_error(stress_num, stress_anal) < RTOL