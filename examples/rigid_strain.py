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

# magnet parameters
msat = 1.5e6
aex = 10.5e-12
alpha = 0.02
K = 7e3

# Elastic coupling constant
B = -84e6
B1 = B
B2 = B

# amplitude, frequency and wave vector of the strain
E, w, k = 105e-6, 1.3e9 * 2*np.pi, 2*np.pi / 2.8e-6

# simulation grid parameters
nx, ny, nz = 2000, 250, 1
cx, cy, cz = 4e-9, 4e-9, 3e-9

# create a world and a magnet
cellsize = (cx, cy, cz)
grid = Grid((nx, ny, nz))
world = World(cellsize)
magnet = Ferromagnet(world, grid) #, geometry=shape.Ellipse(nx*cx, ny*cy).translate(nx*cx/2 - cx/2, ny*cy/2 - cy/2, 0))
magnet.enable_demag = True

# setting magnet parameters
magnet.msat = msat
magnet.aex = aex
magnet.alpha = alpha
magnet.ku1 = K
magnet.anisU = (1,0,0)
magnet.B1 = B1
magnet.B2 = B2

# To set the strain we need time and space functions
# Different strain components are given by
# exx = E sin(kx - wt)
# eyy = 0
# ezz = -E sin(kx - wt)
# exy = 0
# exz = E cos(kx - wt)
# eyz = 0
# These can all be rewritten by using sum and difference formulas
# By using add_time_term we can create a f(x,y,z,t) = h(x,y,z)*g(t) function
# By then adding a second time function we can recreate the effect of the sin and cos

# Splitting everything in single components for the strain results in 8 functions
def norm_strain_xyz_sin(x,y,z):
    exx = E*np.sin(k*x)
    eyy = 0
    ezz = -E*np.sin(k*x)
    return (exx, eyy, ezz)

def norm_strain_xyz_cos(x,y,z):
    exx = -E*np.cos(k*x)
    eyy = 0
    ezz = E*np.cos(k*x)
    return (exx, eyy, ezz)

def norm_strain_t_cos(t):
    exx = np.cos(w*t)
    eyy = 0
    ezz = np.cos(w*t)
    return (exx, eyy, ezz)

def norm_strain_t_sin(t):
    exx = np.sin(w*t)
    eyy = 0
    ezz = np.sin(w*t)
    return (exx, eyy, ezz)


def shear_strain_xyz_cos(x,y,z):
    exy = 0
    exz = E*np.cos(k*x)
    eyz = 0
    return (exy, exz, eyz)

def shear_strain_xyz_sin(x,y,z):
    exy = 0
    exz = E*np.sin(k*x)
    eyz = 0
    return (exy, exz, eyz)

def shear_strain_t_cos(t):
    exy = 0
    exz = np.cos(w*t)
    eyz = 0
    return (exy, exz, eyz)

def shear_strain_t_sin(t):
    exy = 0
    exz = np.sin(w*t)
    eyz = 0
    return (exy, exz, eyz)

# normal stain
# Create the first time term f(x,y,z) = h(x,y,z)*g(t)
magnet.rigid_norm_strain.add_time_term(norm_strain_t_cos, mask=norm_strain_xyz_sin)
# Add a second time term to obtain sin and cos
magnet.rigid_norm_strain.add_time_term(norm_strain_t_sin, mask=norm_strain_xyz_cos)

# shear strain
# Create the first time term f(x,y,z) = h(x,y,z)*g(t)
magnet.rigid_shear_strain.add_time_term(shear_strain_t_cos, mask=shear_strain_xyz_cos)
# Add a second time term to obtain sin and cos
magnet.rigid_shear_strain.add_time_term(shear_strain_t_sin, mask=shear_strain_xyz_sin)

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
ax2.set_title("Final magnetization")
ax2.set_xlabel("x (µm)")
ax2.set_ylabel("y (µm)")

plt.show()

# plot DW position in function of time
plt.plot(np.linspace(0,run,steps)*1e9, (DW_pos)*1e6)
plt.xlabel("time (ns)")
plt.ylabel("Domain wall position (µm)")
plt.show()