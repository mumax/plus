import numpy as np
import pytest
from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.config import neelskyrmion

RTOL = 1e-6

def relative_error(result, wanted):
    return np.abs((wanted - result) / result)

@pytest.fixture(scope="module")  # reuse across tests
def magnet():
    """Return a semi realistic system to test"""
    nx, ny, nz = 128, 128, 1
    cx, cy, cz = 1e-9, 1e-9, 1e-9

    # PBC to avoid boundary torque
    world = World((cx, cy, cz), pbc_repetitions=(1, 1, 0), mastergrid=Grid((nx, ny, 0)))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))

    magnet.msat = 580e3
    magnet.aex = 15e-12
    magnet.ku1 = 0.8e6
    magnet.anisU = (0, 0, 1)
    magnet.alpha = 0.2
    magnet.dmi_tensor.set_interfacial_dmi(3.5e-3)
    magnet.thermal_seed = 1234567  # consistent test

    return magnet

# === Magnet states ===
def set_skyrmion_state(magnet: Ferromagnet):
    """
    Non-trivial, non-random magnetization. Close to optimal, but purpousfully
    not minimized to have higher torque
    """
    magnet.magnetization = neelskyrmion(magnet.center, 20e-9, -1, 1)

def set_uniform_state(magnet: Ferromagnet):
    """This state results in no torque at all."""
    magnet.magnetization = (0, 0, 1)

# === Noise and time step getters ===
def consistent_max_noise(magnet):
    magnet.reset_noise_generator()
    thermal_noise = magnet.thermal_noise.eval()
    return np.max(np.linalg.norm(thermal_noise, axis=0))

def consistent_sensible_timestep(magnet: Ferromagnet):
    magnet.reset_noise_generator()
    return magnet.world.timesolver.sensible_timestep


class TestSensibleTimestep:
    def test_default(self, magnet):
        set_uniform_state(magnet)  # no torque
        magnet.temperature = 0  # no noise

        timesolver = magnet.world.timesolver
        assert timesolver.sensible_timestep == timesolver.sensible_timestep_default

    def test_only_torque(self, magnet):
        set_skyrmion_state(magnet)
        magnet.temperature = 0  # no noise

        factor = magnet.world.timesolver.sensible_factor
        max_torque = magnet.max_torque()

        sens_dt_mumax = consistent_sensible_timestep(magnet)
        assert relative_error(sens_dt_mumax, factor / max_torque) < RTOL
        
    def test_torque_and_noise(self, magnet):
        """Torque and noise are of about equal importance."""
        set_skyrmion_state(magnet)  # torque
        magnet.temperature = 0.1  # some noise
        #  now max_noise^2 / (factor * max_torque) is in the order of magnitude of 1

        max_noise = consistent_max_noise(magnet)
        factor = magnet.world.timesolver.sensible_factor
        max_torque = magnet.max_torque()

        sens_dt_mumax = consistent_sensible_timestep(magnet)
        sens_dt = ((np.sqrt(max_noise**2 + 4*factor*max_torque) - max_noise) / (2 * max_torque))**2
        assert relative_error(sens_dt_mumax, sens_dt) < RTOL

    def test_high_noise_limit(self, magnet):
        """Very high noise limit, but torque still exists."""
        set_skyrmion_state(magnet)  # some torque
        magnet.temperature = 1e3  # a lot of noise

        sens_dt_mumax = consistent_sensible_timestep(magnet)
        sens_dt = (magnet.world.timesolver.sensible_factor / consistent_max_noise(magnet))**2
        assert relative_error(sens_dt_mumax, sens_dt) < RTOL

    def test_only_noise(self, magnet):
        set_uniform_state(magnet)  # no torque
        magnet.temperature = 100  # any amount of noise

        sens_dt_mumax = consistent_sensible_timestep(magnet)
        sens_dt = (magnet.world.timesolver.sensible_factor / consistent_max_noise(magnet))**2
        assert relative_error(sens_dt_mumax, sens_dt) < RTOL
