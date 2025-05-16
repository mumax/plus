"""In this example we move a domain wall by setting a time and space dependent
   strain in a ferromagnet to simulate the effect of a SAW wave. This is
   based on the method used in
   https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.104420."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util import twodomain

# simulation time
run = 10e-9
steps = 1000
dt = run/steps

# simulation grid parameters
nx, ny, nz = 256, 512, 1
cx, cy, cz = 2.4e-9, 2.4e-9, 1e-9

# create a world and a magnet
cellsize = (cx, cy, cz)
grid = Grid((nx, ny, nz))
world = World(cellsize)
magnet = Ferromagnet(world, grid)

# setting magnet parameters
magnet.msat = 6e5
magnet.aex = 1e-11
magnet.alpha = 0.01
magnet.ku1 = 8e5
magnet.anisU = (0,0,1)

# setting DMI to stabilize the DW
magnet.dmi_tensor.set_interfacial_dmi(1e-3)

# Create a DW
magnet.magnetization = twodomain((0,0,1), (-1,0,0), (0,0,-1), nx*cx/3, 5*cx)

print("minimizing...")
magnet.minimize()  # minimize

# magnetoelastic coupling constants
magnet.B1 = -1.5e7
magnet.B2 = 0

# amplitude, angular frequency and wave vector of the strain
E = 6e-3
w = 200e6 * 2*np.pi
k = 4000 / w

# normal stain, given by exx = E [sin(wt)*cos(kx) - cos(wt)*sin(kx)]
# Create the first time term
magnet.rigid_norm_strain.add_time_term(lambda t: (np.sin(w*t), 0., 0.),
                                       lambda x,y,z: (E*np.cos(k*x), 0., 0.))
# Add the second time term
magnet.rigid_norm_strain.add_time_term(lambda t: (np.cos(w*t), 0., 0.),
                                       lambda x,y,z: (-E*np.sin(k*x), 0., 0.))

# plot the initial and final magnetization
fig, axs = plt.subplots(nrows=1, ncols=2, sharex="all", sharey="all")
ax1, ax2 = axs
im_extent = (-0.5*cx*1e6, (nx*cx - 0.5*cx)*1e6, -0.5*cy*1e6, (ny*cy - 0.5*cy)*1e6)

# initial magnetization
ax1.imshow(np.transpose(magnet.magnetization.get_rgb()[:,0,:,:], axes=(1,2,0)), origin="lower", extent=im_extent, aspect="equal")
ax1.set_title("Initial magnetization")
ax2.set_title("Final magnetization")
ax1.set_xlabel("x (µm)")
ax2.set_xlabel("x (µm)")
ax1.set_ylabel("y (µm)")

# function to estimate the position of the DW
def DW_position(magnet):
    m_av = magnet.magnetization.average()[2]
    return m_av*nx*cx / 2 + nx*cx/2

# run the simulation and save the DW postion
DW_pos = np.zeros(shape=(steps+1))
DW_pos[0] = DW_position(magnet)
time = np.zeros(shape=(steps+1))
time[0] = world.timesolver.time

print("running...")
for i in tqdm(range(1, steps+1)):
    world.timesolver.run(dt)
    DW_pos[i] = DW_position(magnet)
    time[i] = world.timesolver.time
print("done!")

# final magnetization
ax2.imshow(np.transpose(magnet.magnetization.get_rgb()[:,0,:,:], axes=(1,2,0)),
           origin="lower", extent=im_extent, aspect="equal")

plt.show()

# plot DW position in function of time
plt.plot(time*1e9, DW_pos*1e6)
plt.xlabel("Time (ns)")
plt.ylabel("Domain wall position (µm)")
plt.title("Domain wall position in time")
plt.show()
