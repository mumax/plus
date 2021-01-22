from mumax3 import Mumax3Simulation
import numpy as np

from mumax5 import Ferromagnet, Grid, World

RTOL = 1e-3


class TestBulkDMI:
    """Test bulk dmi related quantities against mumax3."""

    def setup_class(self):
        # arbitrarily chosen parameters
        msat, bdmi = 800e3, 3e-3
        cellsize = (1e-9, 2e-9, 3.2e-9)
        gridsize = (30, 16, 4)

        self.mumax3sim = Mumax3Simulation(
            f"""
                setcellsize{cellsize}
                setgridsize{gridsize}
                msat = {msat}
                Dbulk = {bdmi}

                m = randommag()
                saveas(m, "m.ovf")
                
                // default in mumax3 is false, mumax5 uses open bc
                openbc = true 

                // The dmi is included in the exchange in mumax3
                // because Aex is set to zero here, b_exch is the dmi field
                saveas(b_exch, "b_dmi.ovf")      
                saveas(edens_exch, "edens_dmi.ovf") 
                saveas(e_exch, "e_dmi.ovf")
            """
        )

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))
        self.magnet.msat = msat
        self.magnet.bdmi = bdmi
        self.magnet.enable_demag = False
        self.magnet.magnetization.set(self.mumax3sim.get_field("m"))

    def test_bulkdmi_field(self):
        wanted = self.mumax3sim.get_field("b_dmi")
        result = self.magnet.bulkdmi_field()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_bulkdmi_in_effective_field(self):
        wanted = self.magnet.bulkdmi_field()
        result = self.magnet.effective_field()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_bulkdmi_energy_density(self):
        wanted = self.mumax3sim.get_field("edens_dmi")
        result = self.magnet.bulkdmi_energy_density()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_bulkdmi_in_total_energy_density(self):
        wanted = self.magnet.bulkdmi_energy_density()
        result = self.magnet.total_energy_density()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_bulkdmi_energy(self):
        wanted = self.mumax3sim.get_field("e_dmi").flat[0]  # same value in all cells
        result = self.magnet.bulkdmi_energy()
        assert np.isclose(result, wanted, rtol=RTOL)

    def test_bulkdmi_in_total_energy(self):
        wanted = self.magnet.bulkdmi_energy()
        result = self.magnet.total_energy()
        assert np.isclose(result, wanted, rtol=RTOL)
