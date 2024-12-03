"""This is the script used to generate the antiferromagnetic dispertion
relation plots in the paper 
"mumax⁺: extensible GPU-accelerated micromagnetics and beyond"
https://arxiv.org/abs/2411.18194
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mumaxplus import World, Grid, Antiferromagnet
import os.path
from mumaxplus.util.constants import *
import matplotlib

font = {'family' : 'serif',
        'size'   : 7}

matplotlib.rc('font', **font)
plt.rc('text', usetex=True)


# angle between magnetization and wave propagation
theta = np.pi/6  # np.pi/2

# magnet parameters
msat = 566e3
a = 0.61e-9
aex = 2.48e-12
A_c = -9.93e5 * a**2
A_nn = 0
K = 611e3
alpha = 2e-3
Bdc = 2

# magnetoelastic parameters
rho = 2800
B = -55e6
B1 = B
B2 = B
c11 = 200e9
c44 = 70e9
c12 = c11 - 2*c44  # assume isotropic
eta = 2e11

# time settings
fmax = 5e12/(2*np.pi)        # maximum frequency (in Hz) of the sinc pulse
time_max = 8e-10             # simulation time (longer -> better frequency resolution)
dt = 1 / (2 * fmax)          # optimal sample time
nt = 1 + int(time_max / dt)  # number of time points

# simulation grid parameters
nx, ny, nz = 4096, 1, 1
# cellsize should stay smaller than exchange length
# and much smaller than the smallest wavelength
cx, cy, cz = 1e-9, 1e-9, 1e-9

# file names
m_filename = "magnon-phonon_magnetizations_AFM_pi6.npy"
u_filename = "magnon-phonon_displacements_AFM_pi6.npy"

def simulation(theta):
    # create a world and a 1D magnet with PBC in x and y
    cellsize = (cx, cy, cz)
    grid = Grid((nx, ny, nz))
    world = World(cellsize, mastergrid=Grid((nx,0,0)), pbc_repetitions=(2,0,0))
    magnet = Antiferromagnet(world, grid)

    magnet.enable_demag = False

    # magnet parameters without magnetoelastics
    for sub in magnet.sublattices:
        sub.msat = msat
        sub.aex = aex
        sub.alpha = alpha
        sub.ku1 = K
        sub.anisU = (np.cos(theta), np.sin(theta), 0)
    magnet.afmex_cell = A_c
    magnet.afmex_nn = A_nn
    magnet.latcon = a
    magnet.sub1.magnetization = (np.cos(theta), np.sin(theta), 0)
    magnet.sub2.magnetization = (-np.cos(theta), -np.sin(theta), 0)
    magnet.bias_magnetic_field = (Bdc*np.cos(theta), Bdc*np.sin(theta), 0)

    print("Minimizing...")
    magnet.minimize()
    print("Minimized!")

    # elastic parameters
    magnet.enable_elastodynamics = True
    for sub in magnet.sublattices:
        sub.B1 = B1
        sub.B2 = B2
    magnet.rho = rho
    magnet.c11 = c11 
    magnet.c44 = c44
    magnet.c12 = c12

    # no displacement initially
    magnet.elastic_displacement = (0, 0, 0)

    # damping
    magnet.sub1.alpha = alpha
    magnet.sub2.alpha = alpha
    magnet.eta = eta

    # time stepping
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = dt/100

    # parameters to save
    m = np.zeros(shape=(nt, 3, nz, ny, nx))
    u = np.zeros(shape=(nt, 3, nz, ny, nx))
    
    # add magnetic field and external force excitation in the middle of the magnet
    Fac = 1e16  # force pulse strength
    Bac = 1e3  # magnetic pulse strength

    mask = np.zeros(shape=(1, 1, nx))
    # Put signal at the center of the simulation box
    mask[:, :, nx // 2 - 1:nx // 2 + 1] = 1
    Fac_dir = np.array([Fac, Fac, Fac])/np.sqrt(3)
    Bac_dir = np.array([Bac, Bac, Bac])/np.sqrt(3)
    def time_force_field(t):
        sinc = np.sinc(2 * fmax * (t - time_max / 2))
        return tuple(sinc * Fac_dir)
    def time_magnetic_field(t):
        sinc = np.sinc(2 * fmax * (t - time_max / 2))
        return tuple(sinc * Bac_dir)
    magnet.external_body_force.add_time_term(time_force_field, mask=mask)
    magnet.bias_magnetic_field.add_time_term(time_magnetic_field, mask=mask)

    # run a while and save the data
    m[0,...] = magnet.sub1.magnetization.eval()
    u[0,...] = magnet.elastic_displacement.eval()
    for i in tqdm(range(1, nt)):
        world.timesolver.run(dt)
        m[i,...] = magnet.sub1.magnetization.eval()
        u[i,...] = magnet.elastic_displacement.eval()
        
    np.save(m_filename, m)
    np.save(u_filename, u)
    return m, u

# check if the files already exist
if os.path.isfile(m_filename):
    m = np.load(m_filename)
    u = np.load(u_filename)
else:
    m, u = simulation(theta)

# plotting ranges
xmin, xmax = 0.01, 0.5  # rad/nm
ymin, ymax = 0.2/(2*np.pi), 2.5/(2*np.pi)  # THz

# x- and y-coordinates of FT cell centers
ks = np.fft.fftshift(np.fft.fftfreq(nx, cx) * 2*np.pi)
fs = np.fft.fftshift(np.fft.fftfreq(nt, dt))
# FT cell widths
dk = ks[1] - ks[0]
df = fs[1] - fs[0]
# image extent of k-values and frequencies, compensated for cell-width
extent = [(ks[0] - dk/2) * 1e-9, (ks[-1] + dk/2) * 1e-9,
          (fs[0] - df/2) * 1e-12, (fs[-1] + df/2) * 1e-12]

# normalize them in the relevant area, so they are visible in the plot
x_start = int((xmin - extent[0]) / (extent[1] - extent[0]) * nx)
x_end = int((xmax - extent[0]) / (extent[1] - extent[0]) * nx)
y_start = int((ymin - extent[2]) / (extent[3] - extent[2]) * nt)
y_end = int((ymax - extent[2]) / (extent[3] - extent[2]) * nt)

# Fourier in time and x-direction of displacement and magnetization
u_FT = np.zeros((3, nt, nx))
m_FT = np.zeros((3, nt, nx))

for i in range(3):
    u_i = np.abs(np.fft.fftshift(np.fft.fft2(u[:,i,0,0,:])))
    u_i_max = np.max(u_i[y_start:y_end, x_start:x_end])
    u_i /= u_i_max
    u_FT[i,...] = u_i

    m_i = np.abs(np.fft.fftshift(np.fft.fft2(m[:,i,0,0,:])))
    m_i_max = np.max(m_i[y_start:y_end, x_start:x_end])
    m_i /= m_i_max
    m_FT[i,...] = m_i

FT_tot = u_FT[0,...] + u_FT[2,...] + m_FT[2,...]

fig, ax = plt.subplots(figsize=(2.5, 4.8/6.4 * 2.5))
linewidth = 1
ls = 'solid'

ax.imshow(FT_tot, aspect='auto', origin='lower', extent=extent,
           vmin=0, vmax=1, cmap="inferno")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Wavenumber (rad/nm)")
ax.set_ylabel("Frequency (THz)")
plt.tight_layout()
plt.savefig("AFMEL_dispersion.pdf", dpi=1200)

# numerical calculations
lambda_exch = (2*aex) / (MU0*msat**2)
k = np.linspace(xmin*1e9, xmax*1e9, 2000)

# elastic waves
vt = np.sqrt(c44/rho)
vl = np.sqrt(c11/rho)
w_t = np.abs(vt*k)
w_l = np.abs(vl*k)
ax.plot(k*1e-9, w_t/(2*np.pi)*1e-12, color="red", lw=linewidth, ls=ls, label="El. trans.")
ax.plot(k*1e-9, w_l/(2*np.pi)*1e-12, color="darkorange", lw=linewidth, ls=ls, label="El. long.")
ax.plot(k*1e-9, -w_t/(2*np.pi)*1e-12, color="red", lw=linewidth, ls=ls)
ax.plot(k*1e-9, -w_l/(2*np.pi)*1e-12, color="darkorange", lw=linewidth, ls=ls)

# spin waves
w_ext = GAMMALL * Bdc
w_ani = GAMMALL * 2*K / msat
w_ex = GAMMALL * 2*aex/msat * k**2
w_c = GAMMALL * 4*A_c/(a**2 * msat)
w_nn = GAMMALL * A_nn/msat * k**2

# Magnon waves
w_mag = np.sqrt((w_ani + w_ex - w_nn)*(w_ani + w_ex - 2*w_c + w_nn))
omega_magn1 = w_mag + w_ext
omega_magn2 = w_mag - w_ext
omega_magn3 = -w_mag + w_ext
omega_magn4 = -w_mag - w_ext
ax.plot(k*1e-9, omega_magn1/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls, label="Mag.")
ax.plot(k*1e-9, omega_magn2/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)
ax.plot(k*1e-9, omega_magn3/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)
ax.plot(k*1e-9, omega_magn4/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)

# Magnon-Phonon waves
J = GAMMALL * B**2 / (rho*msat)
w = np.linspace(ymin*(2*np.pi)*1e12, ymax*(2*np.pi)*1e12, 2000)
k, w = np.meshgrid(k,w)

w_g = w_ani + w_ex - 2*w_c + w_nn
omega_mp = 2*J*k**2 * w_g*np.cos(theta)**2 *(J*k**2 * w_g*(w_l**2+w_t**2-2*w**2)\
                      - (w_l**2-w**2)*(w_t**2-w**2)*(w_mag**2 - w_ext**2-w**2)\
                      + J*k**2 * w_g*(w_l**2-w_t**2)*np.cos(4*theta))

omega_mp += (w_t**2-w**2)*(-J*k**2 * w_g*(w_l**2+w_t**2-2*w**2)*(w_mag**2-w_ext**2-w**2)\
            +(w_l**2-w**2)*(w_t**2-w**2)*(w_mag**2-(w_ext-w)**2)*(w_mag**2-(w_ext+w)**2)\
            -J*k**2 * w_g*(w_l**2-w_t**2)*(w_mag**2-w_ext**2-w**2)*np.cos(4*theta))

contour = ax.contour(k*1e-9, w/(2*np.pi)*1e-12, omega_mp, [0], colors="white",
                     linewidths=0.5, linestyles="dashed")

ax.plot([], [], ls="--", lw=linewidth, label="Mag. el.", color="white")  # for legend entry

# plot numerical result
ax.imshow(FT_tot, aspect='auto', origin='lower', extent=extent,
           vmin=0, vmax=1, cmap="inferno")

# plot cleanup
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Wavenumber (rad/nm)")
ax.set_ylabel("Frequency (THz)")
ax.legend(loc="upper left", fontsize="5")
plt.tight_layout()
plt.savefig("AFMEL_dispersion_theory.pdf", dpi=1200)
