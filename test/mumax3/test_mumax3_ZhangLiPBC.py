from mumax3 import Mumax3Simulation
import numpy as np

from mumaxplus import Ferromagnet, Grid, World


class TestAverage:
    def test_average_magnetization(self):
        nx, ny, nz = 256, 32, 1
        c = 5e-9
        PBC = (1, 0, 0)
        gridsize = (nx, ny, nz)

        msat = 800e3
        aex = 13e-12
        alpha1 = 3
        alpha2 = 0.01
        xi = 0.1
        J = 1e12
        pol = 1

        run = 1e-9

        mumax3sim = Mumax3Simulation(
            f"""
                    setPBC{PBC}
                    setGridSize{gridsize}
                    c := {c}
                    setCellSize(c, c, c)

                    Msat  = {msat}
                    Aex   = {aex}
                    alpha = {alpha1}
                    EnableDemag = False

                    m = twodomain(1,0,0,  0,1,0,  -1,0,0)
                    m.setInShape(xrange(-inf, -120*c), uniform(0,-1,0))
                    saveas(m, "m.ovf")

                    run({run})

                    alpha = {alpha2}
                    xi    = {xi}
                    J     = vector({J}, 0, 0)
                    Pol   = {pol}

                    run({run})
                    saveas(m, "m_final.ovf")
                """
        )

        world = World(
            cellsize=(c, c, c), pbc_repetitions=PBC, mastergrid=Grid((nx, 0, 0))
        )
        magnet = Ferromagnet(world, Grid(gridsize))

        magnet.msat = msat
        magnet.aex = aex
        magnet.alpha = alpha1
        magnet.enable_demag = False

        magnet.magnetization.set(mumax3sim.get_field("m"))

        world.timesolver.run(run)

        magnet.alpha = alpha2
        magnet.xi = xi
        magnet.jcur = (J, 0, 0)
        magnet.pol = pol

        world.timesolver.run(run)

        m_final = magnet.magnetization.eval()
        m_final_mx3 = mumax3sim.get_field("m_final")

        assert np.allclose(m_final, m_final_mx3, atol=1e-3)
