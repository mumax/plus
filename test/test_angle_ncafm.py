import numpy as np
import pytest

from mumaxplus import Antiferromagnet, Ferromagnet, NcAfm, Grid, World

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

class TestMaxAngleNcAfm:

    def test_same_sublattices(self):
        magnet = NcAfm(World((1, 1, 1)), Grid((1, 1, 1)))
        with pytest.raises((ValueError)):
            angle = magnet.max_intracell_angle_between(magnet.sub1, magnet.sub1)

    def test_different_host(self):
        magnet1 = NcAfm(World((1, 1, 1)), Grid((1, 1, 1)))
        magnet2 = NcAfm(World((1, 1, 1)), Grid((1, 1, 1)))
        with pytest.raises((ValueError)):
            angle = magnet1.max_intracell_angle_between(magnet1.sub1, magnet2.sub2)

    def test_afm_sublattice(self):
        magnet1 = NcAfm(World((1, 1, 1)), Grid((1, 1, 1)))
        magnet2 = Antiferromagnet(World((1, 1, 1)), Grid((1, 1, 1)))
        with pytest.raises((ValueError)):
            angle = magnet1.max_intracell_angle_between(magnet1.sub1, magnet2.sub1)
    
    def test_fm(self):
        magnet1 = NcAfm(World((1, 1, 1)), Grid((1, 1, 1)))
        magnet2 = Ferromagnet(World((1, 1, 1)), Grid((1, 1, 1)))
        with pytest.raises((ValueError)):
            angle = magnet1.max_intracell_angle_between(magnet1.sub1, magnet2)

    def test_90(self):
        magnet = NcAfm(World((1, 1, 1)), Grid((1, 1, 1)))
        magnet.sub1.magnetization = (1, 0, 0)
        magnet.sub2.magnetization = (0, 1, 0)
        assert np.isclose(magnet.max_intracell_angle_between(magnet.sub1, magnet.sub2), np.pi / 6)

    def test_0(self):
        magnet = NcAfm(World((1, 1, 1)), Grid((1, 1, 1)))
        magnet.sub1.magnetization = (1, 0, 0)
        magnet.sub2.magnetization = (1, 0, 0)
        assert np.isclose(magnet.max_intracell_angle_between(magnet.sub1, magnet.sub2), 2 * np.pi / 3)

    def test_180(self):
        magnet = NcAfm(World((1, 1, 1)), Grid((1, 1, 1)))
        magnet.sub1.magnetization = (1, 0, 0)
        magnet.sub2.magnetization = (-1, 0, 0)
        assert np.isclose(magnet.max_intracell_angle_between(magnet.sub1, magnet.sub2), np.pi / 3)

    def test_max_angle(self):
        magnet = NcAfm(World((1, 1, 1)), Grid((10, 10, 10)))

        m1 = magnet.sub1.magnetization()
        m2 = magnet.sub2.magnetization()
        m3 = magnet.sub3.magnetization()

        n12 = np.max(np.abs(np.arccos(np.sum(m1 * m2, axis=0)) - 2 * np.pi/3))
        n13 = np.max(np.abs(np.arccos(np.sum(m1 * m3, axis=0)) - 2 * np.pi/3))
        n23 = np.max(np.abs(np.arccos(np.sum(m2 * m3, axis=0)) - 2 * np.pi/3))

        a12 = magnet.max_intracell_angle_between(magnet.sub1, magnet.sub2)
        a13 = magnet.max_intracell_angle_between(magnet.sub1, magnet.sub3)
        a23 = magnet.max_intracell_angle_between(magnet.sub2, magnet.sub3)

        assert np.isclose(n12, a12)
        assert np.isclose(n13, a13)
        assert np.isclose(n23, a23)
    
    def test_angle_field(self):
        magnet = NcAfm(World((1, 1, 1)), Grid((10, 10, 10)))
        magnet.msat = 1
        magnet.ncafmex_cell = -1

        m1 = magnet.sub1.magnetization()
        m2 = magnet.sub2.magnetization()
        m3 = magnet.sub3.magnetization()

        n12 = np.arccos(np.sum(m1 * m2, axis=0)) - 2*np.pi/3
        n13 = np.arccos(np.sum(m1 * m3, axis=0)) - 2*np.pi/3
        n23 = np.arccos(np.sum(m2 * m3, axis=0)) - 2*np.pi/3

        wanted = np.stack([n12, n13, n23], axis=0)
        result = magnet.angle_field()
        assert max_semirelative_error(result, wanted) < 1e-5