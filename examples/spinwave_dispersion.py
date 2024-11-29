import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util.constants import GAMMALL

# Ferromagnetic spinwave dispersion relation

# Disregarding dipole-dipole interactions, the dispersion relation f(k) in a ferromagnetic wire is given by

# f(k) = (gamma / 2 * pi) * ((2A / M_s) * k^2 + B)

# with A the exchange stiffness, M_s the saturation magnetization, and B an externally applied field.
# We will try to reproduce this dispersion relation numerically using mumax.

# The mumax script below simulates a uniformly magnetized nanowire with an applied field perpendicular to the wire.
# Spin waves are excited by a sinc pulse with a maximum frequency of 20GHz at the center of the simulation box.
# Note that the demagnetization is disabled in this simulation.

# Numerical parameters
fmax = 20E9          # maximum frequency (in Hz) of the sinc pulse
T = 1E-8             # simulation time (longer -> better frequency resolution)
dt = 1 / (2 * fmax)  # the sample time
dx = 4E-9            # cellsize
nx = 1024            # number of cells

# Material/system parameters
Bz = 0.2           # bias field along the z direction
A = 13E-12         # exchange constant
Ms = 800E3         # saturation magnetization
alpha = 0.05       # damping parameter
gamma = GAMMALL    # gyromagnetic ratio

# Create the world
grid_size = (nx, 1, 1)
cell_size = (dx, dx, dx)

world = World(cell_size)
world.bias_magnetic_field = (0, 0, Bz)

# Create a ferromagnet
magnet = Ferromagnet(world, Grid(size=grid_size))
magnet.enable_demag = False
magnet.msat = Ms
magnet.aex = A
magnet.alpha = alpha

Bt = lambda t: (0.01 * np.sinc(2 * fmax * (t - T / 2)), 0, 0)
mask = np.zeros(shape=(1, 1, nx))
# Put signal at the center of the simulation box
mask[:, :, 511:513] = 1
magnet.bias_magnetic_field.add_time_term(Bt, mask)

magnet.magnetization = (0, 0, 1)
magnet.minimize()

nt = 1 + int(T / dt)
timepoints = np.linspace(0, T, nt)
outputquantities = {
    'm': lambda: magnet.magnetization.eval(),
    'mx_mean': lambda: magnet.magnetization.average()[0],
    'my_mean': lambda: magnet.magnetization.average()[1],
    'mz_mean': lambda: magnet.magnetization.average()[2],
}

# Run solver
output = world.timesolver.solve(timepoints, outputquantities)

# Plot output data
t = output['time']
mx_mean = output['mx_mean']
my_mean = output['my_mean']
mz_mean = output['mz_mean']

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 8))
axs[0].plot(t, mx_mean)
axs[0].set_ylabel('$M_x$')

axs[1].plot(t, my_mean)
axs[1].set_ylabel('$M_y$')

axs[2].plot(t, mz_mean)
axs[2].set_ylabel('$M_z$')
axs[2].set_xlabel('$t$')

plt.suptitle('Ferromagnetic Spinwave Dispersion Relation (Python)')
plt.show()

# Apply the two dimensional FFT
m = np.array(output['m'], dtype=float)
mx = m[:, 0, 0, 0, :]
mx_fft = np.fft.fft2(mx)
mx_fft = np.fft.fftshift(mx_fft)

plt.figure(figsize=(10, 6))

# x- and y-coordinates of FT cell centers
ks = np.fft.fftshift(np.fft.fftfreq(nx, dx) * 2*np.pi)
fs = np.fft.fftshift(np.fft.fftfreq(nt, dt))
# FT cell widths
dk = ks[1] - ks[0]
df = fs[1] - fs[0]
# image extent of k-values and frequencies, compensated for cell-width
extent = [ks[0] - dk/2, ks[-1] + dk/2, fs[0] - df/2, fs[-1] + df/2]

# Show the intensity plot of the 2D FFT
plt.imshow(np.abs(mx_fft)**2, extent=extent,
           aspect='auto', origin='lower', cmap="inferno")

# Plot the analytical derived dispersion relation
k = np.linspace(-2e8, 2e8, 1000)
freq_theory = A * gamma * k**2 / (np.pi * Ms) + gamma * Bz / (2 * np.pi)
plt.plot(k, freq_theory, 'r--', lw=1)
plt.axhline(gamma * Bz / (2 * np.pi), c='g', ls='--', lw=1)

plt.xlim([-2e8, 2e8])
plt.ylim([0, fmax])
plt.ylabel("$f$ (Hz)")
plt.xlabel("$k$ (1/m)")
plt.title('Ferromagnetic Spinwave Dispersion Relation (Python)')

plt.show()
