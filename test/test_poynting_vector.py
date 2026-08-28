import numpy as np

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
nx, ny, nz = 128, 64, 1  # number of cells
msat = 800e3
C11, C12, C44 = 283e9, 58e9, 166e9
vel = 4

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

def test_poynting():
    """Create a random elastic magnet and test the calculation of
    the poynting vector.
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

    def velocity_func(x, y, z):
        return tuple(np.random.rand(3))

    magnet.elastic_displacement = displacement_func
    magnet.elastic_velocity = velocity_func
    
    stress = magnet.stress_tensor.eval()
    v = magnet.elastic_velocity.eval()

    poynting_num = magnet.poynting_vector.eval()
    poynting_anal = np.zeros(shape=poynting_num.shape)

    poynting_anal[0,...] = stress[0,...] * v[0] + stress[3,...] * v[1] + stress[4,...] * v[2]
    poynting_anal[1,...] = stress[3,...] * v[0] + stress[1,...] * v[1] + stress[5,...] * v[2]
    poynting_anal[2,...] = stress[4,...] * v[0] + stress[5,...] * v[1] + stress[2,...] * v[2]
    poynting_anal *= -1

    assert max_semirelative_error(poynting_num, poynting_anal) < RTOL