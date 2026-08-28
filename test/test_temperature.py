import numpy as np
from mumaxplus import Ferromagnet, Grid, World

class TestTemperature:
    @staticmethod
    def generate_noise(seed=None, grid_size=(16, 16, 4)):
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid(grid_size))
        magnet.msat = 1e3
        magnet.alpha = 1
        magnet.temperature = 10
        if seed is not None:
            magnet.thermal_seed = seed
        return magnet.thermal_noise()

    def test_default_seed(self):
        noise1 = self.generate_noise()
        noise2 = self.generate_noise()
        assert not np.allclose(noise1, noise2)

    def test_set_same_seed(self):
        noise1 = self.generate_noise(1234567)
        noise2 = self.generate_noise(1234567)
        assert np.allclose(noise1, noise2)

    def test_set_different_seed(self):
        noise1 = self.generate_noise(1234567)
        noise2 = self.generate_noise(7654321)
        assert not np.allclose(noise1, noise2)

    def test_grid_compatibility(self):
        """ The CUDA RNG is only compatible with an even number of grid cells.
        Internally, an additional cell is added in the noise generation when
        this is not the case. This function tests if the last cell is cropped
        accordingly. This simultaneously tests the compatibility with an odd
        number of grid cells."""
        noise1 = self.generate_noise(1234567, (10, 1, 1))
        noise2 = self.generate_noise(1234567, (9, 1, 1))
        assert np.allclose(noise1[..., :-1], noise2)

    def test_reset_generator(self):
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((10, 10, 10)))
        magnet.msat = 1e3
        magnet.alpha = 1
        magnet.temperature = 10
        magnet.thermal_seed = 12345

        noise_initial = magnet.thermal_noise()
        magnet.thermal_seed = 12345
        noise_after_seed_reset = magnet.thermal_noise()

        magnet.reset_noise_generator()
        noise_after_RNG_reset = magnet.thermal_noise()

        # Resetting the seed does not reset the RNG
        assert not np.allclose(noise_initial, noise_after_seed_reset)
        assert np.allclose(noise_initial, noise_after_RNG_reset)