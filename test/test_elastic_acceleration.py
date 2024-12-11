import pytest
import numpy as np
import math

import matplotlib.pyplot as plt

from mumaxplus import Grid, World, Ferromagnet


SRTOL = 1e-3

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
nx, ny, nz = 256, 128, 1
N = 1  # PBC
C11 = 283e9
C44 = 58e9
C12 = 166e9
B1 = -8.8e6
B2 = -4.4e6


def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


class TestForces:
    def setup_class(self):
        """Makes a world with a magnet with random elastic parameters.
        """
        world = World(cellsize)

        self.magnet =  Ferromagnet(world, Grid((nx,ny,nz)))
        self.magnet.enable_elastodynamics = True

        self.magnet.eta = np.random.rand(1, nz,ny,nx)
        self.magnet.rho = np.random.rand(1, nz,ny,nx)

        self.magnet.C11 = C11
        self.magnet.C12 = C12
        self.magnet.C44 = C44

        self.magnet.B1 = B1
        self.magnet.B2 = B2

        def displacement_func(x, y, z):
            return tuple(np.random.rand(3) * 1e-15)

        def velocity_func(x, y, z):
            return (np.random.rand(), np.random.rand(), np.random.rand())
        
        self.magnet.elastic_displacement = displacement_func
        self.magnet.elastic_velocity = velocity_func

        self.magnet.external_body_force = np.random.rand(3, nz,ny,nx)

    def test_effective_force(self):
        """Check if the effective body force is the sum of the other forces."""
        force = self.magnet.elastic_force.eval() + self.magnet.external_body_force.eval() + self.magnet.magnetoelastic_force.eval()
        assert max_semirelative_error(self.magnet.effective_body_force.eval(), force) < SRTOL

    def test_damping_calc(self):
        """Check if damping is correctly calculated."""
        damping = - self.magnet.eta.eval() * self.magnet.elastic_velocity.eval()
        assert max_semirelative_error(self.magnet.elastic_damping.eval(), damping) < SRTOL

    def test_damping_add(self):
        """Check if damping is correctly added to the forces. This also checks the division by rho."""
        damping = self.magnet.elastic_damping.eval()
        forces = self.magnet.effective_body_force.eval()
        rho = self.magnet.rho.eval()
        assert max_semirelative_error(rho*self.magnet.elastic_acceleration.eval(), (forces + damping)) < SRTOL
