# This script solves micromagnetic standard problem 5. The Problem specification
# can be found on https://www.ctcms.nist.gov/~rdm/mumag.org.html

import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World

def get_initial_config(position, Radius):
    x0, y0, _ = position

    def func(x, y, z):
        x -= x0
        y -= y0
        return (-y, x, Radius)  # automatically normalized later

    return func

# world specifications
length, width, thickness = 100e-9, 100e-9, 10e-9
nx, ny, nz = 50, 50, 5  # following mumax³ paper
world = World(cellsize=(length / nx, width / ny, thickness / nz))

# parameters
magnet = Ferromagnet(world, Grid((nx, ny, nz)))
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.1

# initial magnetization
Radius = 10e-9
magnet.magnetization = get_initial_config((length/2, width/2, thickness/2), Radius)
magnet.minimize()

# add current
xi1, xi2, xi3, xi4 = 0.0, 0.05, 0.1, 0.5
magnet.xi = xi2  # choose xi here
magnet.pol = 1.0  # purely polarized current
magnet.jcur = (1e12, 0, 0)

# --- SCHEDULE THE OUTPUT ---

timepoints = np.linspace(0, 5e-9, 1000)
outputquantities = {
    "mx": lambda: magnet.magnetization.average()[0],
    "my": lambda: magnet.magnetization.average()[1],
    "mz": lambda: magnet.magnetization.average()[2]
}

# --- RUN THE SOLVER ---

output = world.timesolver.solve(timepoints, outputquantities)

# --- PLOT THE OUTPUT DATA ---

# time series
t_fig, t_ax = plt.subplots()
for key in ["mx", "my", "mz"]:
    t_ax.plot(np.asarray(output["time"])*1e9, output[key], label=key)
t_ax.legend()
t_ax.set_xlabel("t (ns)")
t_ax.set_ylabel("<m>")

#  mx versus my
mm_fig, mm_ax = plt.subplots()
mm_ax.axis("equal")
mm_ax.plot(output["mx"], output["my"])
mm_ax.set_xlabel("$m_x$")
mm_ax.set_ylabel("$m_y$")
mm_ax.grid()

plt.show()
