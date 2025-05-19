import numpy as np
import pytest
from mumax3 import Mumax3Simulation
from mumaxplus import Ferromagnet, Grid, World

RTOL = 1e-3

@pytest.fixture(scope="class", params=[True, False])
def simulations(request):
    openbc = request.param

    # arbitrarily chosen parameters
    msat, dbulk, aex = 800e3, 3e-3, 10e-12
    cellsize = (1e-9, 2e-9, 3.2e-9)
    gridsize = (30, 16, 4)

    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{cellsize}
            setgridsize{gridsize}
            msat = {msat}
            aex = {aex}
            Dbulk = {dbulk}

            m = randommag()
            saveas(m, "m.ovf")
            
            // default in mumax³ is false, these tests use both
            openbc = {openbc}

            // The dmi is included in the exchange in mumax³
            // because Aex is set to zero here, b_exch is the dmi field
            saveas(b_exch, "b_exch_dmi.ovf")      
            saveas(edens_exch, "edens_exch_dmi.ovf") 
            saveas(e_exch, "e_exch_dmi.ovf")
        """
    )

    world = World(cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))
    magnet.enable_demag = False
    magnet.enable_openbc = openbc
    magnet.msat = msat
    magnet.aex = aex
    magnet.dmi_tensor.set_bulk_dmi(dbulk)
    magnet.magnetization.set(mumax3sim.get_field("m"))

    return world, magnet, mumax3sim


@pytest.mark.mumax3
class TestBulkDMI:
    """Test bulk dmi against mumax³."""

    def test_dmi_field(self, simulations):
        world, magnet, mumax3sim = simulations
        wanted = mumax3sim.get_field("b_exch_dmi")
        result = np.add(magnet.dmi_field(), magnet.exchange_field())
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_dmi_in_effective_field(self, simulations):
        world, magnet, mumax3sim = simulations
        wanted = np.add(magnet.dmi_field(), magnet.exchange_field())
        result = magnet.effective_field()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_dmi_energy_density(self, simulations):
        world, magnet, mumax3sim = simulations
        wanted = mumax3sim.get_field("edens_exch_dmi")
        result = np.add(magnet.dmi_energy_density(),
                        magnet.exchange_energy_density())
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_dmi_in_total_energy_density(self, simulations):
        world, magnet, mumax3sim = simulations
        wanted = np.add(magnet.dmi_energy_density(),
                        magnet.exchange_energy_density())
        result = magnet.total_energy_density()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_dmi_energy(self, simulations):
        world, magnet, mumax3sim = simulations
        wanted = mumax3sim.get_field("e_exch_dmi").flat[0]  # same value in all cells
        result = np.add(magnet.dmi_energy(), magnet.exchange_energy())
        assert np.isclose(result, wanted, rtol=RTOL)

    def test_dmi_in_total_energy(self, simulations):
        world, magnet, mumax3sim = simulations
        wanted = np.add(magnet.dmi_energy(), magnet.exchange_energy())
        result = magnet.total_energy()
        assert np.isclose(result, wanted, rtol=RTOL)
