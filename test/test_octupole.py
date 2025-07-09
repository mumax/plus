import numpy as np
from mumaxplus import NcAfm, Grid, World

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

def compute_octupole_vector(m1, m2, m3, ms1, ms2, ms3):

    def rotate_120(m, ref):
        k = np.cross(ref, m, axis=0)
        k /= np.linalg.norm(k, axis=0)
        s = np.sign(np.sum(ref * m, axis=0))
        s[s == 0] = 1 # perpendicular vectors
        return -0.5 * m + np.cross(k, m, axis=0) * (np.sqrt(3) / 2) * s

    m2r = rotate_120(m2, m1)
    m3r = rotate_120(m3, m1)

    return (m1 * ms1 + m2r * ms2 + m3r * ms3) / (ms1 + ms2 + ms3)

class TestOctupoleVector:
    def test_octupole_vector(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((32, 32, 32)))
        ms1, ms2, ms3 = 123, 456, 789
        magnet.sub1.msat = ms1
        magnet.sub2.msat = ms2
        magnet.sub3.msat = ms3
        m1 = magnet.sub1.magnetization()
        m2 = magnet.sub2.magnetization()
        m3 = magnet.sub3.magnetization()

        result = magnet.octupole_vector()
        wanted = compute_octupole_vector(m1, m2, m3, ms1, ms2, ms3)
        assert max_semirelative_error(result, wanted) < 5e-6

    def test_octupole_vector_uniform(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((1, 1, 1)))
        magnet.msat = 10
        magnet.magnetization = (1, 0, 0)

        result = magnet.octupole_vector()
        assert result.all() == 0

    def test_octupole_vector_120(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((1, 1, 1)))
        magnet.msat = 10
        for i, sub in enumerate(magnet.sublattices):
            theta = i * 120 * np.pi / 180
            sub.magnetization = (np.cos(theta), np.sin(theta), 0)

        result = magnet.octupole_vector()
        assert np.allclose(result.squeeze(), [1, 0, 0])