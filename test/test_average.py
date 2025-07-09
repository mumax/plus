import numpy as np

from mumaxplus import Ferromagnet, Grid, World



class TestAverage:
    def test_average_magnetization(self):
        world = World((1, 1, 1))
        magnet = Ferromagnet(world, Grid((32, 32, 5), (-10, 3, 0)))
        m = magnet.magnetization.get()
        wanted = np.average(m, axis=(1, 2, 3))
        result = magnet.magnetization.average()
        for i in range(3):
            assert np.abs((wanted[i] - result[i]) / result[i]) < 1e-5

    def test_average_zeeman_energy_geometry(self):
        gridsize = (4, 5, 6)
        geo = np.zeros(gridsize[::-1])
        geo[:3, :, :] = 1 # geometry spans half of grid

        w = World((1, 1, 1))
        m1 = Ferromagnet(w, Grid(gridsize))
        m2 = Ferromagnet(w, Grid(gridsize, origin=(0, 0, gridsize[2])), geometry=geo)
        m1.magnetization, m2.magnetization = (1, 0, 1), (1, 0, 1)

        w.bias_magnetic_field = (0, 0, 1)

        assert np.isclose(0.5 * m1.zeeman_energy(), m2.zeeman_energy(), rtol=1e-12, atol=1e-12)
