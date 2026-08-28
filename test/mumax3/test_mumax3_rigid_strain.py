import numpy as np
import pytest
from mumax3 import Mumax3Simulation
from mumaxplus import Ferromagnet, Grid, World

RTOL = 3e-5
strain = 1e-4

def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

def simulation(exx, eyy, ezz, exy, exz, eyz):
    # arbitrarily chosen parameters
    msat, aex, alpha = 566e3, 2.48e-12, 0.02
    B = -55e6
    cellsize = (1e-9, 2e-9, 3.2e-9)
    gridsize = (30, 16, 4)

    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{cellsize}
            setgridsize{gridsize}
            msat = {msat}
            aex = {aex}
            alpha = {alpha}
            exx = {exx}
            eyy = {eyy}
            ezz = {ezz}
            exy = {exy}
            exz = {exz}
            eyz = {eyz}
            B1 = {B}
            B2 = {B}

            m = randommag()
            saveas(m, "m.ovf")
            saveas(B_mel, "B.ovf")
        """
    )

    world = World(cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))
    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha
    magnet.magnetization.set(mumax3sim.get_field("m"))
    magnet.rigid_norm_strain = (exx, eyy, ezz)
    magnet.rigid_shear_strain = (exy, exz, eyz)
    magnet.B1 = B
    magnet.B2 = B

    return magnet, mumax3sim


@pytest.mark.mumax3
class TestRigidStrain:
    """Test rigid strain against mumax³."""
    def test_exx(self):
        exx, eyy, ezz, exy, exz, eyz = strain, 0, 0, 0, 0, 0
        magnet, mumax3sim = simulation(exx, eyy, ezz, exy, exz, eyz)
        wanted = mumax3sim.get_field("B")
        result = magnet.magnetoelastic_field.eval()
        err = max_relative_error(result, wanted)
        assert err < RTOL

    def test_eyy(self):
        exx, eyy, ezz, exy, exz, eyz = 0, strain, 0, 0, 0, 0
        magnet, mumax3sim = simulation(exx, eyy, ezz, exy, exz, eyz)
        wanted = mumax3sim.get_field("B")
        result = magnet.magnetoelastic_field.eval()
        err = max_relative_error(result, wanted)
        assert err < RTOL

    def test_ezz(self):
        exx, eyy, ezz, exy, exz, eyz = 0, 0, strain, 0, 0, 0
        magnet, mumax3sim = simulation(exx, eyy, ezz, exy, exz, eyz)
        wanted = mumax3sim.get_field("B")
        result = magnet.magnetoelastic_field.eval()
        err = max_relative_error(result, wanted)
        assert err < RTOL

    def test_exy(self):
        exx, eyy, ezz, exy, exz, eyz = 0, 0, 0, strain, 0, 0
        magnet, mumax3sim = simulation(exx, eyy, ezz, exy, exz, eyz)
        wanted = mumax3sim.get_field("B")
        result = magnet.magnetoelastic_field.eval()
        print(np.any(wanted != 0))
        print(np.any(result != 0))
        err = max_relative_error(result, wanted)
        assert err < RTOL

    def test_exz(self):
        exx, eyy, ezz, exy, exz, eyz = 0, 0, 0, 0, strain, 0
        magnet, mumax3sim = simulation(exx, eyy, ezz, exy, exz, eyz)
        wanted = mumax3sim.get_field("B")
        result = magnet.magnetoelastic_field.eval()
        err = max_relative_error(result, wanted)
        assert err < RTOL

    def test_eyz(self):
        exx, eyy, ezz, exy, exz, eyz = 0, 0, 0, 0, 0, strain
        magnet, mumax3sim = simulation(exx, eyy, ezz, exy, exz, eyz)
        wanted = mumax3sim.get_field("B")
        result = magnet.magnetoelastic_field.eval()
        err = max_relative_error(result, wanted)
        assert err < RTOL

    def test_combined(self):
        exx, eyy, ezz = -strain, -strain, -strain
        exy, exz, eyz = -strain, -strain, -strain
        magnet, mumax3sim = simulation(exx, eyy, ezz, exy, exz, eyz)
        wanted = mumax3sim.get_field("B")
        result = magnet.magnetoelastic_field.eval()
        err = max_relative_error(result, wanted)
        assert err < RTOL