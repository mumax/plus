import numpy as np

import matplotlib.pyplot as plt

from mumaxplus import Grid, World, Antiferromagnet


SRTOL = 1e-5

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
nx, ny, nz = 256, 128, 1

msat = 1.2e6
B1 = -4.4e6
B2 = -8.8e6

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


def check_magnetoelastic_field(host, sublattice):
    strain = host.strain_tensor.eval()
    
    B_num = sublattice.magnetoelastic_field.eval()
    B_anal = np.zeros(shape=B_num.shape)

    m = sublattice.magnetization.eval()
    for i in range(3):
        ip1 = (i+1)%3
        ip2 = (i+2)%3

        B_anal[i,...] = - 2  / msat * (
            B1 *  strain[i,...] * m[i,...] + 
            B2 * (strain[i+ip1+2,...] * m[ip1,...] + 
                  strain[i+ip2+2,...] * m[ip2,...]))

    assert max_semirelative_error(B_num, B_anal) < SRTOL
    


class TestMelAfm:

    def setup_class(self):
        world = World(cellsize)

        self.magnet = Antiferromagnet(world, Grid((nx,ny,nz)))
        self.magnet.msat = msat

        self.magnet.enable_elastodynamics = True

        self.magnet.C11 = 283e9
        self.magnet.C12 = 58e9
        self.magnet.C44 = 166e9

        # magnetoelasticity for both sublattices
        self.magnet.B1 = B1
        self.magnet.B2 = B2

        self.magnet.elastic_displacement = 1e-12 * np.random.random((3, nz, ny, nx))

        self.magnet.elastic_velocity = np.random.random((3, nz, ny, nx))

        for sub in self.magnet.sublattices:
            sub.magnetization = np.random.random((3, nz, ny, nx))

        
    def test_effective_force(self):
        """Tests total effective body force to make sure it contains
        contributions of both sublattices.
        """
        force = self.magnet.elastic_force.eval() + \
                self.magnet.sub1.magnetoelastic_force.eval() + \
                self.magnet.sub2.magnetoelastic_force.eval()
        assert max_semirelative_error(self.magnet.effective_body_force.eval(), force) < SRTOL

    def test_different_magnetoelastic_force(self):
        """Make sure the magnetoelastic forces are different (analytical model
        is too difficult).
        """
        assert not np.all(self.magnet.sub1.magnetoelastic_force.eval() ==
                          self.magnet.sub2.magnetoelastic_force.eval())

    def test_magnetoelastic_field_sub1(self):
        """Check magnetoelastic field of sublattice 1."""
        check_magnetoelastic_field(self.magnet, self.magnet.sub1)

    def test_magnetoelastic_field_sub2(self):
        """Check magnetoelastic field of sublattice 2."""
        check_magnetoelastic_field(self.magnet, self.magnet.sub2)

    def test_total_energy_energy(self):
        """Make sure the sum of energies is correct."""
        energy_density = (self.magnet.sub1.total_energy_density.eval() +
                          self.magnet.sub2.total_energy_density.eval() +
                          self.magnet.kinetic_energy_density.eval() +
                          self.magnet.elastic_energy_density.eval())

        assert max_semirelative_error(self.magnet.total_energy_density.eval(),
                                      energy_density) < SRTOL
