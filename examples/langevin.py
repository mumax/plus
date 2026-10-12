import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World


@np.vectorize
def expectation_mz_langevin(msat, bext, temperature, cellvolume):
    kB = 1.381e-23
    xi = msat * cellvolume * bext / (kB * temperature)
    return 1 / np.tanh(xi) - 1 / xi


msat = 800e3
bext = 0.05
cellvolume = 100e-27
temperatures = np.linspace(1, 500, 25)

N = 1024
relaxtime = 2e-9
sampletime = 1e-9
nsamples = 200

world = World(cellsize=3 * [np.power(cellvolume, 1.0 / 3.0)])
world.bias_magnetic_field = (0, 0, bext)
magnet = Ferromagnet(world, Grid((N, 1, 1)))
magnet.enable_demag = False
magnet.aex = 0.0
magnet.alpha = 0.1
magnet.msat = msat
magnet.magnetization = (0, 0, 1)  # groundstate

m_simul = []
for temperature in temperatures:
    magnet.temperature = temperature

    world.timesolver.run(relaxtime)

    timepoints = np.linspace(world.timesolver.time, world.timesolver.time + sampletime, nsamples)
    outputquantities = {"mz": lambda: magnet.magnetization.average()[2]}
    output = world.timesolver.solve(timepoints, outputquantities)

    m_simul.append(np.average(output["mz"]))

m_langevin = expectation_mz_langevin(msat, bext, temperatures, cellvolume)

plt.plot(temperatures, m_simul, "o", label="Simulation")
plt.plot(temperatures, m_langevin, "k-", label="Theory")
plt.xlabel("Temperature (K)")
plt.ylabel(r"$<m_z>$")
plt.legend()
plt.show()
