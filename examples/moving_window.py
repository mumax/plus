"""
In this example we move a domain wall in a ferromagnet using a Zhang-Li STT. We let the simulation
window move together with the wall, keeping the domain wall centered in the simulation space.
Using this, we can virtually simulate an infitly long magnetic nanowire using a limited number
of simulation cells.

Note:
The moving window functionality only works properly if
- The magnet parameters are uniform
- The magnet has no geometry
- The magnet has no regions
"""
from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util import twodomain, plot_field

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

# ----------- Material and simulation parameters -----------
cs = 1e-9
length = 256e-9
width = 64e-9
thickness = 1e-9

Ms = 600e3
aex = 10e-12
alpha = 0.02
ku = 6e5
anisU = (1, 0, 0)

# ----------- Create magnet -----------
world = World((cs, cs, cs))
grid = Grid((int(length / cs), int(width / cs), int(thickness / cs)))

magnet = Ferromagnet(world, grid)

magnet.msat  = Ms
magnet.aex   = aex
magnet.alpha = alpha
magnet.ku1   = ku
magnet.anisU = anisU

magnet.enable_demag = False
magnet.enable_openbc = True


# ----------- Create two domain state -----------
magnet.magnetization = twodomain((1, 0, 0), (0, 1, 0), (-1, 0, 0), magnet.center[0], 5e-9)
magnet.minimize()

# ----------- Add current and simulate -----------
magnet.jcur = (-1e12, 0, 0)
magnet.xi   = 0.2
magnet.pol = 1

# Center the simulation window, keeping component 0 (x) close to zero.
# We expect motion alongt he x axis.
world.center_domain_wall(comp=0, axis=0)

tmax = 2e-9
timepoints = np.linspace(0, tmax, 100)
outputquantities = {"mag": lambda: magnet.magnetization(),
                    "pos": lambda: world.window.position[0]}

output = world.timesolver.solve(timepoints, outputquantities, tqdm=True)

# ----------- Create movie -----------
print("Creating animation...")
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# Time trace subplot
ax_trace = axes[0]
lines = {}
lines["pos"], = ax_trace.plot([], [], '-')

ax_trace.set_title("Domain wall position")
ax_trace.set_xlim(0, tmax * 1e9)
ax_trace.set_ylim(min(0, min(output["pos"]) * 1e9), max(0, max(output["pos"]) * 1e9))
ax_trace.set_xlabel("Time $t$ (ns)")
ax_trace.set_ylabel("position (nm)")
ax_trace.grid()

# Magnetization image subplot
ax_image = axes[1]
plot_field(output["mag"][0], ax=ax_image, arrow_size=8)

ax_image.set_xlim(0, int(length/cs))
ticks = ax_image.get_xticks()
ticks = ticks[ticks <= int(length/cs)] # remove matplotlibs invisible tick at 300

ax_image.set_title("$t$ = 0.000 ns")
ax_image.set_xlabel("$x$ (nm)")
ax_image.set_ylabel("$y$ (nm)")

fig.tight_layout()

# --- Animation Function ---
def update(frame):
    # Update image
    ax_image.clear()
    plot_field(output["mag"][frame], ax=ax_image, arrow_size=8)
    ax_image.set_xlabel("$x$ (nm)")
    ax_image.set_ylabel("$y$ (nm)")

    # update x-ticks
    ax_image.set_xticks(ticks)
    ax_image.set_xticklabels([f"{t + output["pos"][frame] * 1e9:.0f}" for t in ticks])
    ax_image.set_title(f"$t$ = {output['time'][frame] * 1e9:.3f} ns")

    # Update time trace
    lines["pos"].set_data(np.array(output["time"][:frame+1]) * 1e9, np.array(output["pos"][:frame+1]) * 1e9)
    return [ax_image] + list(lines.values())

# Animation parameters
fps = 15
anim = FuncAnimation(fig, update, frames=len(output["time"]),
                     interval=1000 / fps, repeat_delay=5000 / fps)

# --- Save the animation ---
save_filename = "moving_window.mp4"
writer = FFMpegWriter(fps=fps)
anim.save(save_filename, writer=writer)