import numpy as np
import matplotlib.pyplot as plt
from mumaxplus.util.show import plot_field
from mumaxplus import Ferromagnet, Grid, World

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(np.linalg.norm(wanted, axis=0))

msat = 5.8e5
aex = 10e-12
K = 7e5
K2 = 1.5e5
alpha = 0.1

B = -8.8e6
B1 = B
B2 = B

C11 = 283e9
C12 = 166e9
C44 = 58e9
rho = 8e3

nx, ny, nz = 256, 256, 1
cx, cy, cz = 1e-9, 1e-9, 1e-9

cellsize = (cx, cy, cz)
grid = Grid((nx, ny, nz))

def simulation(t=None):
    world = World(cellsize)
    magnet = Ferromagnet(world, grid)

    magnet.enable_elastodynamics = True
    magnet.magnetization = (0,0,1)

    # magnet parameters without magnetoelastics
    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha
    magnet.ku1 = K
    magnet.ku2 = K2
    magnet.anisU = (0,0,1)

    magnet.C11 = C11
    magnet.C12 = C12
    magnet.C44 = C44
    magnet.rho = rho

    magnet.B1 = B1
    magnet.B2 = B2

    def disp(x,y,z):
        return (0,1e-9 * np.sin(x),0)

    magnet.elastic_displacement = disp

    world.timesolver.timestep = 1e-13

    print(t)
    if t != None:
        world.timesolver.adaptive_timestep = False
        world.timesolver.timestep = t

    world.timesolver._impl.run(0.1e-9)

    return magnet.elastic_displacement.eval()

class TestAdaptiveElsatic:
    def test_average_magnetization(self):

        displacement_adapt = simulation()
        displacement_static = simulation(1e-13)

        assert max_semirelative_error(displacement_adapt, displacement_static) < 1e-4
