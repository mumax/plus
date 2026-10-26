"""Two vortices in neighbouring permalloy disks, driven from the browser GUI.

    python examples/vortex_gui.py

then open the printed URL. Press Run, then try:
  * Parameters: change alpha, or bias_magnetic_field to 0, 0, 0.01 (T);
    Region -1 = whole magnet
  * Select disk1 or disk2 to view and edit each magnet separately
  * Solver: switch the Runge-Kutta method, set fixdt, press Relax
  * Initial m: pick "vortex" and press Set
  * Plot: the energies and dmz set up at the bottom of this script; + Add plot for more
  * Console: Run(1e-9)   or   magnet2.magnetization = (1, 0, 0)
"""

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.gui import GUI
from mumaxplus.util import Circle, vortex

nx, ny, nz = 128, 128, 3
cell = 4e-9
world = World(cellsize=(cell, cell, 10e-9))


def make_disk(origin_x):
    """Disk filling a grid whose lower-left cell is at cell index origin_x along x."""
    disk = Circle(nx * cell)
    disk.translate((origin_x + nx / 2) * cell - cell / 2, ny * cell / 2 - cell / 2, 0)
    return disk


gap = 2  # cells between the two grids

magnet = Ferromagnet(world, Grid((nx, ny, nz)), name="disk1", geometry=make_disk(0))
magnet2 = Ferromagnet(world, Grid((nx, ny, nz), origin=(nx + gap, 0, 0)),
                      name="disk2", geometry=make_disk(nx + gap))

for mag in (magnet, magnet2):
    mag.msat = 800e3
    mag.aex = 13e-12
    mag.alpha = 0.01

# opposite circulation in the second disk
magnet.magnetization = vortex(magnet.center, 8e-9, 1, 1)
magnet2.magnetization = vortex(magnet2.center, 8e-9, -1, 1)

# namespace=globals() makes `magnet`, `magnet2` and `world` available in the GUI console
gui = GUI(world, namespace=globals(), open_browser=True)

# what to put in the table (and table.txt), and what to show, before the page opens
gui.table_add(magnet.total_energy, name="E1")
gui.table_add(magnet2.total_energy, name="E2")
gui.table_add(
    lambda: magnet.magnetization.average()[2] - magnet2.magnetization.average()[2],
    name="dmz",
)
gui.table_autosave(1e-12)  # one row in table.txt every ps of simulation time
gui.add_plot("t", "E1")    # the first call replaces the default plot
gui.add_display("magnetization")  # the first call replaces the default display
gui.add_display("effective_field", comp="z")

gui.run()
