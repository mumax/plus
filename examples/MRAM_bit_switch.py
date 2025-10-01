"""This script switches the state of a bit in a vertical MRAM stack (free layer
on the bottom, spacer layer, fixed layer on top) using Slonczewski spin transfer
torque. This is based on both the paper "The design and verification of MuMax3"
https://doi.org/10.1063/1.4899186 and on session 3 example 3 of the MuMax3
workshop https://mumax.ugent.be/mumax3-workshop/ ."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import Ellipse, get_rgb


# --- set up the world ---
length, width, thickness = 160e-9, 80e-9, 5e-9
nx, ny, nz = 64, 32, 1
cx, cy, cz = length/nx, width/ny, thickness/nz
world = World(cellsize=(cx, cy, cz))

# --- set up ferromagnet ---
geometry = Ellipse(length, width)
# translate to the center of the simulation box ((0,0,0) is center of first cell)
geometry.translate(length/2 - cx/2, width/2 - cy/2, thickness/2 - cz/2)

magnet = Ferromagnet(world, Grid((nx, ny, nz)), geometry=geometry)
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.01
magnet.magnetization = (1, 0, 0)
magnet.minimize()

# add polarized current and fixed FM layer
magnet.pol = 0.5669
magnet.Lambda = 2
magnet.epsilon_prime = 1
area = length*width*np.pi/4
magnet.jcur = (0, 0, -4e-3/area)  # -4 mA

theta = 20 * np.pi/180
magnet.fixed_layer = (np.cos(theta), np.sin(theta), 0)


# --- schedule the output ---
tmax = 0.5e-9
timepoints = np.linspace(0, tmax, 151)
outputquantities = {"mx": lambda: magnet.magnetization.average()[0],
                    "my": lambda: magnet.magnetization.average()[1],
                    "mz": lambda: magnet.magnetization.average()[2],
                    "rgb": lambda: get_rgb(magnet.magnetization, layer=0)}
# --- run te solver ---
output = world.timesolver.solve(timepoints, outputquantities)


# --- plot <m> in time ---
fig, ax = plt.subplots(figsize=(8, 5))
for key in ["mx", "my", "mz"]:
    ax.plot(timepoints*1e9, output[key], '-', label=key)

# make plot pretty
ax.set_title("Elliptical MRAM bit switch")
ax.set_xlim(0, tmax*1e9); ax.set_ylim(-1, 1)
ax.set_xlabel("time $t$ (ns)"); ax.set_ylabel("<m>")
ax.legend()
ax.grid()
fig.tight_layout()
plt.show()


# -- animate magnetization rgb ---
fig, ax = plt.subplots(figsize=(6.4, 4))

extent = np.asarray([-0.5*cx, length - 0.5*cx, -0.5*cy, width - 0.5*cy]) * 1e9
im = ax.imshow(output["rgb"][0], origin="lower", extent=extent)
ax.set_xlabel("$x$ (nm)"); ax.set_ylabel("$y$ (nm)")
fig.tight_layout()

def plot_frame(i):
    im.set_data(output["rgb"][i])
    ax.set_title(f"$t$ = {output['time'][i]*1e9:.3f} ns")
    return [im]

fps = 15
repeat_delay = 5000/fps
anim = FuncAnimation(fig, plot_frame, frames=len(output["time"]),
                     interval=1000/fps, repeat_delay=repeat_delay)

# anim.save(filename="MRAM_bit_switch.gif", writer="ffmpeg")
plt.show()
