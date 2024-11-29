"""This example initializes a magnet magnetized to the right.
The magnet is minimized and the elastic parameters are assigned together
with a circular area in which an external sinusoidal field is applied
in the y-direction. This simulation runs for 0.5 ns and returns an animation
of the y-magnetization and the amplified displacement.
The animation might take a while.
"""

import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mumaxplus import World, Grid, Ferromagnet
import mumaxplus.util.shape as shapes


length, width, thickness = 1e-6, 1e-6, 20e-9
nx, ny, nz = 256, 256, 1
cx, cy, cz = length/nx, width/ny, thickness/nz
cellsize = (cx, cy, cz)

grid = Grid((nx, ny, nz))
world = World(cellsize, mastergrid=Grid((nx, ny, 0)), pbc_repetitions=(2, 2, 0))
magnet = Ferromagnet(world, grid)

magnet.msat = 1.2e6
magnet.aex = 18e-12
magnet.alpha = 0.004
Bdc = 5e-3
magnet.bias_magnetic_field = (Bdc, 0, 0)  # uniform term
magnet.magnetization = (1, 0, 0)

magnet.minimize()

magnet.enable_elastodynamics = True
magnet.rho = 8e3
magnet.B1 = -8.8e6
magnet.B2 = -8.8e6
magnet.c11 = 283e9
magnet.c44 = 58e9
magnet.c12 = 166e9

magnet.elastic_displacement = (0, 0, 0)

# add magnetic field excitation
f_B = 9.8e9
Bac = 1e-3
Bdiam = 200e-9  # excitation diameter

# circular magnetic field excitation at the center
Bshape = shapes.Circle(Bdiam).translate(*magnet.center)
Bmask = magnet._get_mask_array(Bshape, grid, world, "mask")
magnet.bias_magnetic_field.add_time_term(
            lambda t: (0, Bac * math.sin(2*math.pi*f_B * t), 0), mask=Bmask)

magnet.alpha = 0.004
magnet.eta = 1e10  # elastic force damping

def displacement_to_scatter_data(magnet, scale, skip):
    """takes magnet.elastic_displacement.eval() array and turns it into an array
    of positions usable by plt.scatter().set_offsets"""

    u = magnet.elastic_displacement.eval()  # ((ux, uy, uz), Z, Y, X)
    coords = magnet.elastic_displacement.meshgrid + scale*u  # absolute positions amplified
    return  np.transpose(coords[:2, 0, ::skip, ::skip].reshape(2, -1))  # to (X*Y, 2)

# plotting
fig, ax = plt.subplots()

u_scale = 5e4  # amplification of displacement
u_skip  = 5  # don't show every displacement

world.timesolver.adaptive_timestep = False
world.timesolver.timestep = 1e-12

steps = 400
time_max = 0.8e-9
duration = time_max/steps

# save magnetization and displacement
m_shape = np.transpose(magnet.magnetization.eval()[1,0,:,:]).shape
u_shape = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip).shape
m = np.zeros(shape=(steps,m_shape[0],m_shape[1]))
u = np.zeros(shape=(steps,u_shape[0],u_shape[1]))

# run a simulation
print("Simulating...")
for i in tqdm(range(steps)):
    world.timesolver.run(duration)
    m[i,...] = np.transpose(magnet.magnetization.eval()[1,0,:,:])
    u[i,...] = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip) 

# scatter setup
offsets = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip)
u_scatter = ax.scatter(offsets[:, 0], offsets[:, 1], s=10, c="black", marker=".", alpha=0.5)

# imshow setup
im_extent = (-0.5*cx, length - 0.5*cx, -0.5*cy, width - 0.5*cy)
vmax, vmin = np.max(m), np.min(m)
vmax = max(abs(vmax), abs(vmin))
vmin = -vmax
m_im = ax.imshow(m[0,...], origin="lower", extent=im_extent, vmin=vmin, vmax=vmax, cmap="seismic")

# colorbar setup
cbar = plt.colorbar(m_im)
cbar.ax.set_ylabel(r"$<m_y>$", rotation=270)

# final touches
ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$y$ (m)")
ax.set_xlim(im_extent[0], im_extent[1])
ax.set_ylim(im_extent[2], im_extent[3])

def update(i):
    m_im.set_data(m[i,...])
    u_scatter.set_offsets(u[i,...])
    return m_im, u_scatter

# animation
print("Animating...")
animation_fig = animation.FuncAnimation(fig, update, frames=steps, interval=40, blit=True)
animation_fig.save("magnetoelastic.mp4")
