from mumax3 import Mumax3Simulation
import numpy as np

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import constants


class TestAverage:
    def test_average_magnetization(self):
        seed = 1234567

        nx, ny, nz = 64, 32, 16
        sx, sy, sz = 128e-9, 64e-9, 32e-9
        gridsize = (nx, ny, nz)
        cellsize = (sx / nx, sy / ny, sz / nz)

        msat = 800e3
        aex = 13e-12
        alpha = 0.01
        lamb = 1
        pol = 0.5669
        epsilonprime = 0.02
        gammaLL = 2.211e5 / constants.MU0
        Jtot = -0.008
        carea = sx * sy
        jc = Jtot / carea
        J = (0, 0, jc)

        grainSize = 4e-9
        maxregion = 256 - 1

        mumax3sim = Mumax3Simulation(
            f"""
                    setsolver(4)
                    randSeed({seed})

                    setgridsize{gridsize}
                    setcellsize{cellsize}

                    Msat = {msat}
                    Aex = {aex}

                    m = randommagseed({seed})

                    alpha = {alpha}
                    Lambda = {lamb}
                    Pol = {pol}
                    epsilonprime = {epsilonprime}
                    DisableZhangLiTorque = true
                    gammaLL = {gammaLL}
                    J = vector{J}

                    maxRegion := {maxregion}
                    ext_makegrains({grainSize}, maxRegion, {seed})

                    for i := 0; i < maxRegion; i++ {{
                        FixedLayer.SetRegion(i, vector(randNorm(), randNorm(), randNorm()))
                    }}

                    saveas(m, "m.ovf")
                    saveas(fixedlayer, "layer.ovf")

                    saveas(STTorque, "STT.ovf")
                """
        )

        world = World(cellsize=cellsize)
        magnet = Ferromagnet(world, Grid(gridsize))

        magnet.msat = msat
        magnet.aex = aex
        magnet.alpha = alpha
        magnet.enable_demag = False
        magnet.enable_zhang_li_torque = False
        magnet.Lambda = lamb
        magnet.pol = pol
        magnet.epsilon_prime = epsilonprime
        magnet.gamma = gammaLL
        magnet.jcur = J

        magnet.magnetization.set(mumax3sim.get_field("m"))
        magnet.fixed_layer.set(mumax3sim.get_field("layer"))

        STT = magnet.spin_transfer_torque.eval() / gammaLL
        STT_mx3 = mumax3sim.get_field("STT")
        assert np.allclose(STT, STT_mx3, rtol=1e-3)
