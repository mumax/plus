import numpy as np

from mumaxplus import Antiferromagnet, NcAfm, Grid, World

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

def compute_local_dmi_field(magnet, sub, symmetry_factor):
    D = np.array(magnet.dmi_vector.average())
    return symmetry_factor / sub.msat.average()[0] * np.cross(D[:, None, None, None],
                                                              sub.magnetization(), axis=0)
class TestLocalDMI:
    def test_local_dmi_afm(self):
        world = World((1, 1, 1))
        magnet = Antiferromagnet(world, Grid((32, 32, 5)))
        magnet.msat = 5
        magnet.dmi_vector = (8, 42, 3e-1)

        for i, sub in enumerate(magnet.sublattices):
            result = sub.homogeneous_dmi_field()
            wanted = compute_local_dmi_field(magnet, magnet.other_sublattice(sub), (-1)**i)
            assert max_semirelative_error(result, wanted) < 1e-6

    def test_local_dmi_ncafm(self):
        world = World((1, 1, 1))
        magnet = NcAfm(world, Grid((32, 32, 5)))
        magnet.msat = 5
        magnet.dmi_vector = (8, 42, 3e-1)

        for i, sub in enumerate(magnet.sublattices):
            result = sub.homogeneous_dmi_field()
            sub2, sub3 = [s for j, s in enumerate(magnet.sublattices) if j != i]
            
            wanted = compute_local_dmi_field(magnet, sub2, (-1)**(i)) + \
                     compute_local_dmi_field(magnet, sub3, (-1)**(i+1))
            assert max_semirelative_error(result, wanted) < 1e-6
