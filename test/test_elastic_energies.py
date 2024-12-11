import numpy as np

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
P = 2  # periods
A = 4  # amplitude
nx, ny, nz = 128, 64, 1  # number of 1D cells
msat = 800e3
B1 = -8.8e6
B2 = B1/2
C11, C12, C44 = 283e9, 58e9, 166e9

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

class TestElasticEnergies:
    def setup_class(self):
        """Makes a world with a magnet with random elasticity parameters
        """
        world = World(cellsize)
        
        self.magnet =  Ferromagnet(world, Grid((nx,ny,nz)))
        self.magnet.enable_elastodynamics = True

        self.magnet.msat = msat

        self.magnet.C11 = C11
        self.magnet.C12 = C12
        self.magnet.C44 = C44

        self.magnet.B1 = B1
        self.magnet.B2 = B2

        def displacement_func(x, y, z):
            return tuple(np.random.rand(3))

        def velocity_func(x, y, z):
            return tuple(np.random.rand(3))

        self.magnet.elastic_displacement = displacement_func
        self.magnet.elastic_velocity = velocity_func
        self.magnet.rho = np.random.rand(1, nz, ny, nx)        

    def test_elastic(self):
        strain_num = self.magnet.strain_tensor.eval()
        stress_num = self.magnet.stress_tensor.eval()
        
        E_el_num = self.magnet.elastic_energy_density.eval()
        
        E_el_anal = np.zeros(shape=E_el_num.shape)
        for i in range(3):
            E_el_anal += 0.5 * stress_num[i,...] * strain_num[i,...] + stress_num[i+3,...] * strain_num[i+3,...]
        
        assert max_semirelative_error(E_el_num, E_el_anal) < RTOL

    def test_kinetic(self):
        v = self.magnet.elastic_velocity.eval()
        rho = self.magnet.rho.eval()

        E_kin_num = self.magnet.kinetic_energy_density.eval()
        E_kin_anal = rho * np.linalg.norm(v, axis=0)**2 / 2
        
        assert max_semirelative_error(E_kin_num, E_kin_anal) < RTOL

    def test_magnetoelastic(self):
        strain = self.magnet.strain_tensor.eval()
        m = self.magnet.magnetization.eval()

        E_num = self.magnet.magnetoelastic_energy_density.eval()
        E_anal = np.zeros(shape=E_num.shape)

        for i in range(3):
            ip1 = (i+1)%3
            ip2 = (i+2)%3

            E_anal += (B1 *  strain[i,...] * m[i,...] * m[i,...] + 
                       B2 * (strain[i+ip1+2,...] * m[ip1,...] * m[i,...] + 
                             strain[i+ip2+2,...] * m[ip2,...] * m[i,...]))

        assert max_semirelative_error(E_num, E_anal) < RTOL