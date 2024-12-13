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
theta = np.pi/2  # np.pi/6 or np.pi/2

# file names
m_filename = f"magnon-phonon_magnetizations_AFM_{int(np.round(theta*180/np.pi))}.npy"
u_filename = f"magnon-phonon_displacements_AFM_{int(np.round(theta*180/np.pi))}.npy"
# output
figname = f"AFMEL_dispersion_theory_{int(np.round(theta*180/np.pi))}.pdf"

plot_mode = "dark"  # "dark or light"

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
C11 = 200e9
C44 = 70e9
C12 = C11 - 2*C44  # assume isotropic
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
    magnet.C11 = C11 
    magnet.C44 = C44
    magnet.C12 = C12

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
# total image extent of k-values and frequencies, compensated for cell-width
totextent = [(ks[0] - dk/2) * 1e-9, (ks[-1] + dk/2) * 1e-9,
          (fs[0] - df/2) * 1e-12, (fs[-1] + df/2) * 1e-12]

# normalize them in the relevant area, so they are visible in the plot
x_start = int(np.round((xmin - totextent[0]) / (totextent[1] - totextent[0]) * nx))
x_end = int(np.round((xmax - totextent[0]) / (totextent[1] - totextent[0]) * nx))
y_start = int(np.round((ymin - totextent[2]) / (totextent[3] - totextent[2]) * nt))
y_end = int(np.round((ymax - totextent[2]) / (totextent[3] - totextent[2]) * nt))

# limited useful image extent
extent = [(ks[x_start] - dk/2) * 1e-9, (ks[x_end] + dk/2) * 1e-9,
          (fs[y_start] - df/2) * 1e-12, (fs[y_end] + df/2) * 1e-12]

# Fourier in time and x-direction of displacement and magnetization
u_FT = np.zeros((3, nt, nx))
m_FT = np.zeros((3, nt, nx))

def hsl_to_rgb(H, S, L):
    """Convert color from HSL to RGB."""
    Hp = np.mod(H/(np.pi/3.0), 6.0)
    C = np.where(L<=0.5, 2*L*S, 2*(1-L)*S)
    X = C * (1 - np.abs(np.mod(Hp, 2.0) - 1.0))
    m = L - C / 2.0

    # R = m + X for 1<=Hp<2 or 4<=Hp<5
    # R = m + C for 0<=Hp<1 or 5<=Hp<6
    R = m + np.select([((1<=Hp)&(Hp<2)) | ((4<=Hp)&(Hp<5)),
                        (Hp<1) | (5<=Hp)], [X, C], 0.)
    # G = m + X for 0<=Hp<1 or 3<=Hp<4
    # G = m + C for 1<=Hp<3
    G = m + np.select([(Hp<1) | ((3<=Hp)&(Hp<4)),
                        (1<=Hp)&(Hp<3)], [X, C], 0.)
    # B = m + X for 2<=Hp<3 or 5<=Hp<6
    # B = m + C for 3<=Hp<5
    B = m + np.select([((2<=Hp)&(Hp<3)) | (5<=Hp),
                        (3<=Hp)&(Hp<5)], [X, C], 0.)

    # clip rgb values to be in [0,1]
    R, G, B = np.clip(R,0.,1.), np.clip(G,0.,1.), np.clip(B,0.,1.)

    return R, G, B


RGBs = []
# IP long u_l, IP trans u_t, OoP trans u_y, m
components = [u[:,0,0,0,:], u[:,1,0,0,:], u[:,2,0,0,:], m[:,2,0,0,:]]
hues = np.array([320, 140, 20, 250]) * np.pi/180
if plot_mode == "light": hues += np.pi
powers = [0.7, 0.8, 0.8, 1.0]
for comp, hue, power in zip(components, hues, powers):
    FT = np.abs(np.fft.fftshift(np.fft.fft2(comp)))[y_start:y_end, x_start:x_end]
    FT_max = np.max(FT)
    print(FT_max)
    FT /= FT_max

    HSL = np.zeros(shape=(3, *FT.shape))
    HSL[0,...] =  hue
    HSL[1,...] = 1  # full saturation
    HSL[2,...] = 0.5 * FT**power  # black to saturated color

    RGB = hsl_to_rgb(*HSL)
    RGB = np.stack(RGB)
    RGBs.append(RGB)

rgb = np.sum(RGBs, axis=0)
if plot_mode == "light":  rgb = np.ones_like(rgb) - rgb
rgb = np.transpose(rgb, (1, 2, 0))

fig, ax = plt.subplots(figsize=(2.5 * 2, 4.8/6.4 * 2.5 * 2))
linewidth = 0.5
lcolor = "black" if plot_mode == "light" else "white"
lalpha=0.5

ls_mag = "-"
ls_long = "-."
ls_trans = ":"
ls_total = "--"

# numerical calculations
lambda_exch = (2*aex) / (MU0*msat**2)
k = np.linspace(xmin*1e9, xmax*1e9, 2000)

# elastic waves
vt = np.sqrt(C44/rho)
vl = np.sqrt(C11/rho)
w_t = np.abs(vt*k)
w_l = np.abs(vl*k)
ax.plot(k*1e-9, w_l/(2*np.pi)*1e-12, color=lcolor, lw=linewidth, ls=ls_long, alpha=lalpha)
ax.plot(k*1e-9, w_t/(2*np.pi)*1e-12, color=lcolor, lw=linewidth, ls=ls_trans, alpha=lalpha)
ax.plot(k*1e-9, -w_t/(2*np.pi)*1e-12, color=lcolor, lw=linewidth, ls=ls_trans, alpha=lalpha)
ax.plot(k*1e-9, -w_l/(2*np.pi)*1e-12, color=lcolor, lw=linewidth, ls=ls_long, alpha=lalpha)

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
ax.plot(k*1e-9, omega_magn1/(2*np.pi)*1e-12, color=lcolor, lw=linewidth, ls=ls_mag, alpha=lalpha)
ax.plot(k*1e-9, omega_magn2/(2*np.pi)*1e-12, color=lcolor, lw=linewidth, ls=ls_mag, alpha=lalpha)
ax.plot(k*1e-9, omega_magn3/(2*np.pi)*1e-12, color=lcolor, lw=linewidth, ls=ls_mag, alpha=lalpha)
ax.plot(k*1e-9, omega_magn4/(2*np.pi)*1e-12, color=lcolor, lw=linewidth, ls=ls_mag, alpha=lalpha)

# Magnon-Phonon waves
J = GAMMALL * B**2 / (rho*msat)
w = np.linspace(ymin*(2*np.pi)*1e12, ymax*(2*np.pi)*1e12, 2000)
k, w = np.meshgrid(k,w)

w_g = w_ani + w_ex - 2*w_c + w_nn

if theta != np.pi/2:  # in general
    omega_mp = 2*J*k**2 * w_g*np.cos(theta)**2 *(J*k**2 * w_g*(w_l**2+w_t**2-2*w**2)\
                        - (w_l**2-w**2)*(w_t**2-w**2)*(w_mag**2 - w_ext**2-w**2)\
                        + J*k**2 * w_g*(w_l**2-w_t**2)*np.cos(4*theta))

    omega_mp += (w_t**2-w**2)*(-J*k**2 * w_g*(w_l**2+w_t**2-2*w**2)*(w_mag**2-w_ext**2-w**2)\
                +(w_l**2-w**2)*(w_t**2-w**2)*(w_mag**2-(w_ext-w)**2)*(w_mag**2-(w_ext+w)**2)\
                -J*k**2 * w_g*(w_l**2-w_t**2)*(w_mag**2-w_ext**2-w**2)*np.cos(4*theta))
else:  # specific reduction
    omega_mp = -2*J*k**2*w_g*(w_mag**2 - w_ext**2 - w**2) + \
                (w_t**2 - w**2)*(w_mag**2 - (w_ext - w)**2)*(w_mag**2 - (w_ext + w)**2)

contour = ax.contour(k*1e-9, w/(2*np.pi)*1e-12, omega_mp, [0], colors=lcolor,
                     linewidths=linewidth, linestyles=ls_total, alpha=lalpha)

# plot numerical result
ax.imshow(rgb, aspect='auto', origin='lower', extent=extent)

# plot cleanup
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Wavenumber (rad/nm)")
ax.set_ylabel("Frequency (THz)")

# --- legend ---
# extra imports
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import legend_handler

labels = ["IP longitudinal elastic", "IP transversal elastic", "OP transversal elastic",
          "Magnetic", "Magnetoelastic"]
linestyles = [ls_long, ls_trans, ls_trans, ls_mag, ls_total]
if plot_mode == "light":
    colors = [hsl_to_rgb(hue + np.pi, 1, 0.5) for hue in hues]
else:
    colors = [hsl_to_rgb(hue, 1, 0.5) for hue in hues]
if plot_mode == "light":
    colors.append((1, 1, 1))  # total has no color
else:
    colors.append((0, 0, 0))  # total has no color
handles = []
for ls, color in zip(linestyles, colors):
    patch = Patch(facecolor=color, edgecolor=color)
    line = Line2D([0], [0], color=lcolor, lw=linewidth, ls=ls)  # , alpha=lalpha)
    handles.append((patch, line))

legend = ax.legend(handles=handles, labels=labels, loc="upper left", fontsize="7",
                   facecolor="white" if plot_mode == "light" else "black", labelcolor=lcolor)
legend.get_frame().set_linewidth(linewidth)
# --- end legend ---

plt.tight_layout()
plt.savefig(figname, dpi=1200)
