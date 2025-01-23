"""In this example we move a domain wall by setting time and space dependant
   strains in a ferromagnet to simulate the effect of a SAW wave. This is
   based on the method used in https://arxiv.org/abs/2406.12778."""

import numpy as np
import matplotlib.pyplot as plt
from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util import twodomain

# simulation time
run = 10e-9
steps = 1000
dt = run/steps

# simulation grid parameters
nx, ny, nz = 2000, 250, 1
cx, cy, cz = 4e-9, 4e-9, 3e-9

# create a world and a magnet
cellsize = (cx, cy, cz)
grid = Grid((nx, ny, nz))
world = World(cellsize)
magnet = Ferromagnet(world, grid)
# setting magnet parameters
magnet.msat = 1.5e6
magnet.aex = 10.5e-12
magnet.alpha = 0.02
magnet.ku1 = 7e3
magnet.anisU = (1,0,0)
B = -84e6
magnet.B1 = B
magnet.B2 = B

# amplitude, angular frequency and wave vector of the strain
E = 105e-6
w = 1.3e9 * 2*np.pi
k = 2*np.pi / 2.8e-6

# To set the strain we need time and space functions
# Different strain components are given by
# exx = E sin(kx - wt)
# eyy = 0
# ezz = -E sin(kx - wt)
# exy = 0
# exz = E cos(kx - wt)
# eyz = 0
# These can all be rewritten by using sum and difference formulas.
# By using add_time_term we can create a f(x,y,z,t) = h(x,y,z)*g(t) function.
# By then adding a second time function we can recreate the effect of the sin and cos.
# Splitting everything in single components for the strain results in 8 functions.

# normal stain
# Create the first time term f(t,x,y,z) = g(t)*h(x,y,z)
magnet.rigid_norm_strain.add_time_term(lambda t: (np.cos(w*t), 0., np.cos(w*t)),
                                       lambda x,y,z: (E*np.sin(k*x), 0., -E*np.sin(k*x)))
# Add a second time term to obtain sin
magnet.rigid_norm_strain.add_time_term(lambda t: (np.sin(w*t), 0., np.sin(w*t)),
                                       lambda x,y,z: (-E*np.cos(k*x), 0., E*np.cos(k*x)))

# shear strain
# Create the first time term f(t,x,y,z) = g(t)*h(x,y,z)
magnet.rigid_shear_strain.add_time_term(lambda t: (0., np.cos(w*t), 0.),
                                        lambda x,y,z: (0., E*np.cos(k*x), 0.))
# Add a second time term to obtain cos
magnet.rigid_shear_strain.add_time_term(lambda t: (0., np.sin(w*t), 0.),
                                        lambda x,y,z: (0., E*np.sin(k*x), 0.))

# Create a domain wall
magnet.magnetization = twodomain((1,0,0), (0,-1,0), (-1,0,0), nx*cx/2, 10*cx)

# plot the initial and final magnetization
fig, ax = plt.subplots(nrows=2, ncols=1)
ax1, ax2 = ax
im_extent = (-0.5*cx*1e6, (nx*cx - 0.5*cx)*1e6, -0.5*cy*1e6, (ny*cy - 0.5*cy)*1e6)

# initial magnetization
ax1.imshow(np.transpose(magnet.magnetization.get_rgb()[:,0,:,:], axes=(1,2,0)), origin="lower", extent=im_extent, aspect="equal")
ax1.set_title("Initial magnetization")
ax2.set_title("Final magnetization")
ax2.set_xlabel("x (µm)")
ax2.set_ylabel("y (µm)")

# function to estimate the position of the DW
def DW_position(magnet):
    m_av = magnet.magnetization.average()[0]
    return m_av*nx*cx / 2

# run the simulation and save the DW postion
DW_pos = np.zeros(shape=(steps))

for i in range(steps):
    world.timesolver.run(dt)
    DW_pos[i] = DW_position(magnet)

# final magnetization
ax2.imshow(np.transpose(magnet.magnetization.get_rgb()[:,0,:,:], axes=(1,2,0)), origin="lower", extent=im_extent, aspect="equal")

plt.show()

# plot DW position in function of time
plt.plot(np.linspace(0,run,steps)*1e9, (DW_pos)*1e6)
plt.xlabel("time (ns)")
plt.ylabel("Domain wall position (µm)")
plt.show()
