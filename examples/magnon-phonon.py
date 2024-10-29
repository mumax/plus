"""This script creates the magnetoelastic dispersion relation when the 
wave propagation and the magnetization form an angle theta as described in
Magnetoelastic Waves in Thin Films.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mumaxplus import World, Grid, Ferromagnet
import os.path
from mumaxplus.util.constants import *

# angle between magnetization and wave propagation
theta = 0

# time settings
dt = 1e-11
time_max = 10e-9
steps = int(time_max / dt)

# simulation grid parameters
length, width, thickness = 20e-6, 30e-9, 30e-9
nx, ny, nz = 4096, 1, 1
cx, cy, cz = length/nx, width/ny, thickness/nz

# magnet parameters
msat = 480e3
aex = 8e-12
alpha = 0.045
Bdc = 50e-3

# magnetoelastic parameters
rho = 8900
B1 = -10e6
B2 = -10e6
c11 = 245e9 
c44 = 75e9
c12 = c11 - 2*c44  # assume isotropic


def simulation(theta):
    # create a world and a 1D magnet with PBC in x and y
    cellsize = (cx, cy, cz)
    grid = Grid((nx, ny, nz))
    world = World(cellsize, mastergrid=Grid((nx,ny,0)), pbc_repetitions=(2,100,0))
    magnet = Ferromagnet(world, grid)

    # magnet parameters without magnetoelastics
    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha
    magnet.magnetization = (np.cos(theta), np.sin(theta), 0)
    magnet.bias_magnetic_field = (Bdc*np.cos(theta), Bdc*np.sin(theta), 0)
    magnet.magnetization = (np.cos(theta), np.sin(theta), 0)
    magnet.bias_magnetic_field = (Bdc*np.cos(theta), Bdc*np.sin(theta), 0)

    magnet.relax()

    # magnetoelastic parameters
    magnet.enable_elastodynamics = True
    magnet.rho = rho
    magnet.B1 = B1
    magnet.B2 = B2
    magnet.c11 = c11 
    magnet.c44 = c44
    magnet.c12 = c12

    # no displacement initially
    magnet.elastic_displacement = (0, 0, 0)

    # add magnetic field excitation
    Bac = 1e-3
    Bdiam = 200e-9
    time_pulse = 20e-12

    # damping
    magnet.alpha = 1e-3
    magnet.eta = 0

    # time stepping
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = 1e-13

    # parameters to save
    m = np.zeros(shape=(steps, 3, nz, ny, nx))
    u = np.zeros(shape=(steps, 3, nz, ny, nx))
    
    # magnetic pulse excitiation in the middle of the magnet
    def pulse_func(x, y, z):
        x -= magnet.center[0]
        if -Bdiam/2 < x < Bdiam/2:
            return (Bdc*np.cos(theta) + Bac*np.sin(theta), Bdc*np.sin(theta) + Bac*np.cos(theta), 0)
        return (Bdc*np.cos(theta), Bdc*np.sin(theta), 0)
    
    magnet.bias_magnetic_field = pulse_func
    
    world.timesolver.run(time_pulse)

    # run a while and save the data
    magnet.bias_magnetic_field = (Bdc*np.cos(theta), Bdc*np.sin(theta), 0)
    for i in tqdm(range(steps)):
        world.timesolver.run(dt)
        m[i,...] = magnet.magnetization.eval()
        u[i,...] = magnet.elastic_displacement.eval()
        
    np.save("m.npy", m)
    np.save("u.npy", u)
    return m, u

# check if the files already exist
if os.path.isfile("m.npy"):
    m = np.load("m.npy")
    u = np.load("u.npy")
else:
    m, u = simulation(theta)

# plotting ranges
xmin, xmax = 2, 20
ymin, ymax = 3, 14
extent = [-(2 * np.pi) / (2 * cx) * (nx+1)/nx * 1e-6,
          (2 * np.pi) / (2 * cx) * (nx-1)/nx * 1e-6,
          -1 / (2 * dt) * steps/(steps-1) * 1e-9,
          1 / (2 * dt) * steps/(steps-1) * 1e-9]

# Fourier in time and x-direction of displacement and magnetization
# normalize all of them in the relevant area, so they are visible in the plot
x_start = int((xmin - extent[0]) / (extent[1] - extent[0]) * nx)
x_end = int((xmax - extent[0]) / (extent[1] - extent[0]) * nx)
y_start = int((ymin - extent[2]) / (extent[3] - extent[2]) * steps)
y_end = int((ymax - extent[2]) / (extent[3] - extent[2]) * steps)

ux_FT = np.abs(np.fft.fftshift(np.fft.fft2(u[:,0,0,0,:])))
ux_FT /= np.max(ux_FT[y_start:y_end,x_start:x_end])

uy_FT = np.abs(np.fft.fftshift(np.fft.fft2(u[:,1,0,0,:])))
uy_FT /= np.max(uy_FT[y_start:y_end,x_start:x_end])

uz_FT = np.abs(np.fft.fftshift(np.fft.fft2(u[:,2,0,0,:])))
uz_FT /= np.max(uz_FT[y_start:y_end,x_start:x_end])

mx_FT = np.abs(np.fft.fftshift(np.fft.fft2(m[:,0,0,0,:])))
mx_FT /= np.max(mx_FT[y_start:y_end,x_start:x_end])

my_FT = np.abs(np.fft.fftshift(np.fft.fft2(m[:,1,0,0,:])))
my_FT /= np.max(my_FT[y_start:y_end,x_start:x_end])

mz_FT = np.abs(np.fft.fftshift(np.fft.fft2(m[:,2,0,0,:])))
mz_FT /= np.max(mz_FT[y_start:y_end,x_start:x_end])

FT_tot = ux_FT + uy_FT + uz_FT + mx_FT + my_FT + mz_FT

# numerical calculations
lambda_exch = (2*aex) / (MU0*msat**2)
k = np.linspace(xmin*1e6, xmax*1e6, 100)

# elastic waves
vt = np.sqrt(c44/rho)
vl = np.sqrt(c11/rho)
omega_t = vt*k
omega_l = vl*k

# spin waves
omega_0 = GAMMALL * Bdc
omega_M = GAMMALL * MU0 * msat
P = 1 - (1 - np.exp(-k*cz)) / (k*cz)
omega_fx = omega_0 + omega_M * (lambda_exch * k**2 + P * np.sin(theta)**2)
omega_fy = omega_0 + omega_M * (lambda_exch * k**2 + 1 - P)

# plot analytical results
plt.plot(k*1e-6, omega_t/(2*np.pi)*1e-9)
plt.plot(k*1e-6, omega_l/(2*np.pi)*1e-9)
plt.plot(k*1e-6, np.sqrt(omega_fx*omega_fy)/(2*np.pi)*1e-9)

# plot numerical result
plt.imshow(FT_tot**2, aspect='auto', origin='lower', extent=extent,
               vmin=0, vmax=1, cmap="inferno")

# plot cleanup
plt.xlim(xmin, xmax)
plt.ylim(ymin,ymax)
plt.xlabel("wavenumber (rad/µm)")
plt.ylabel("wavenumber (GHz)")
plt.show()