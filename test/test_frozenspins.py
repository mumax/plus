from mumaxplus import Grid, World, Ferromagnet

def expectv(result, wanted, tol):
    for i in range(3):
        assert abs(result[i] - wanted[i]) < tol

def test_frozenspins():
    """Based on standard problem 4, but all spins are fixed, so there should be no dynamics."""

    world = World((500e-9/128, 125e-9/32, 3e-9/2))
    magnet = Ferromagnet(world, Grid((128, 32, 2)))

    magnet.msat = 800e3
    magnet.aex = 13e-12
    magnet.alpha = 0.02
    magnet.bias_magnetic_field = (-24.6E-3, 4.3E-3, 0)
    magnet.frozen_spins = 1
    magnet.magnetization = (1, .1, 0)
    
    avg0 = magnet.magnetization.average()

    world.timesolver.run(1e-9)

    expectv(magnet.magnetization.average(), avg0, 1e-7)


def test_frozenspins_inhomogeneous():
    """Frozen spins in half the geometry.
    Compare to vectors to make sure the behaviour did not change as of writing this test.
    """
    world = World((500e-9/128, 125e-9/32, 3e-9/2))
    magnet = Ferromagnet(world, Grid((128, 32, 2)))

    magnet.msat = 800e3
    magnet.aex = 13e-12
    magnet.alpha = 0.02
    magnet.magnetization = (1, .1, 0)

    # freeze right half
    center_x = magnet.center[0]
    magnet.frozen_spins = lambda x, y, z: 1 if x > center_x else 0

    magnet.magnetization = (1, .1, 0)

    # minimize
    magnet.minimize()
    tol = 1e-4
    expectv(magnet.magnetization.average(), (0.9809795618057251, 0.11742201447486877, 0), tol)

    # reversal
    magnet.bias_magnetic_field = (-24.6E-3, 4.3E-3, 0)
    world.timesolver.run(1e-9)
    expectv(magnet.magnetization.average(), (0.14499470591545105, 0.24905619025230408, 0.0021978835575282574), tol)
