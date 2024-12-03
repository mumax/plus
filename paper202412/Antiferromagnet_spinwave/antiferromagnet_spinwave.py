import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util.constants import GAMMALL, MU0
import os.path
import matplotlib
from matplotlib.patches import ConnectionPatch


font = {'family' : 'serif',
        'size'   : 7}

matplotlib.rc('font', **font)
plt.rc('text', usetex=True)

m_big_filename = "AFM_disp.npy"
m_zoom_filename = "AFM_disp_zoom.npy"


# Material/system parameters
a = 0.35e-9              # lattice constant
Bz = 0.4             # bias field along the z direction
A = 10E-12          # exchange constant
A_nn = -5E-12
A_c = -400E-12
Ms = 400e3          # saturation magnetization
K = 1e3

def simulate(fmax, T, dt, nt, dx, dy, dz, nx, alpha, m_filename):
    # Create the world
    grid_size = (nx, 1, 1)
    cell_size = (dx, dy, dz)

    world = World(cell_size)
    world.bias_magnetic_field = (0, 0, Bz)

    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = dt/50

    # Create a ferromagnet
    magnet = Antiferromagnet(world, Grid(size=grid_size))
    magnet.msat = Ms
    magnet.aex = A
    magnet.alpha = alpha
    magnet.ku1 = K
    magnet.anisU = (0, 0, 1)

    magnet.afmex_nn = A_nn
    magnet.afmex_cell = A_c
    magnet.latcon = a

    Bt = lambda t: (1 * np.sinc(2 * fmax * (t - T / 2)), 0, 0)
    mask = np.zeros(shape=(1, 1, nx))
    # Put signal at the center of the simulation box
    mask[:, :, nx // 2 - 1:nx // 2 + 1] = 1
    for sub in magnet.sublattices:
        sub.bias_magnetic_field.add_time_term(Bt, mask)

    magnet.sub1.magnetization = (0, 0, 1)
    magnet.sub2.magnetization = (0, 0, -1)

    # Run solver
    m = np.zeros(shape=(nt, 3, 1, 1, nx))
    m[0,...] = magnet.sub1.magnetization.eval()
    for i in tqdm(range(nt)):
        world.timesolver.run(dt)
        m[i,...] = magnet.sub1.magnetization.eval()
    np.save(m_filename, m)
    return m

# check if the files already exist
# big figure
fmax_big = 5e13
T_big = 2e-12
dt_big = 1 / (2 * fmax_big)    # the sample time
nt_big = 1 + int(T_big / dt_big)
dx_big = 0.5e-9
nx_big = 2560
alpha_big = 1e-3        # damping parameter
if os.path.isfile(m_big_filename):
    m_big = np.load(m_big_filename)
else:
    m_big = simulate(fmax_big, T_big, dt_big, nt_big, dx_big, dx_big, dx_big,
                     nx_big, alpha_big, m_big_filename)

# zoom figure
fmax_zoom = 1e12
T_zoom = 0.2e-9
dt_zoom = 1 / (2 * fmax_zoom)    # the sample time
nt_zoom = 1 + int(T_zoom / dt_zoom)
dx_zoom = 50e-9
dy_zoom, dz_zoom = 0.5e-9, 0.5e-9
nx_zoom = 5120
alpha_zoom = 1e-6        # damping parameter
if os.path.isfile(m_zoom_filename):
    m_zoom = np.load(m_zoom_filename)
else:
    m_zoom = simulate(fmax_zoom, T_zoom, dt_zoom, nt_zoom, dx_zoom, dy_zoom,
                      dz_zoom, nx_zoom, alpha_zoom, m_zoom_filename)

# everything for the main plot

# plotting ranges
xmin, xmax = -1.25, 1.25  # rad/nm
ymin, ymax = 0, 50  # THz

# x- and y-coordinates of FT cell centers
ks = np.fft.fftshift(np.fft.fftfreq(nx_big, dx_big) * 2*np.pi)
fs = np.fft.fftshift(np.fft.fftfreq(nt_big, dt_big))
# FT cell widths
dk = ks[1] - ks[0]
df = fs[1] - fs[0]
# image extent of k-values and frequencies, compensated for cell-width
extent = [(ks[0] - dk/2) * 1e-9, (ks[-1] + dk/2) * 1e-9,
          (fs[0] - df/2) * 1e-12, (fs[-1] + df/2) * 1e-12]

# normalize them in the relevant area, so they are visible in the plot
x_start = int((xmin - extent[0]) / (extent[1] - extent[0]) * nx_big)
x_end = int((xmax - extent[0]) / (extent[1] - extent[0]) * nx_big)
y_start = int((ymin - extent[2]) / (extent[3] - extent[2]) * nt_big)
y_end = int((ymax - extent[2]) / (extent[3] - extent[2]) * nt_big)

# Fourier in time and x-direction of displacement and magnetization
m_FT = np.zeros((3, nt_big, nx_big))

for i in range(3):
    m_big_i = np.abs(np.fft.fftshift(np.fft.fft2(m_big[:,i,0,0,:])))
    m_big_max_i = np.max(m_big_i[y_start:y_end, x_start:x_end])
    m_big_i /= m_big_max_i
    m_FT[i,...] = m_big_i

FT_tot = m_FT[1,...]

# numerical calculations
k = np.linspace(xmin*1e9, xmax*1e9, 2000)

fig, ax = plt.subplots(figsize=(2.5, 4.8/6.4 * 2.5))
linewidth = 1
ls = '--'

# spin waves
w_ext = GAMMALL * Bz
w_ani = GAMMALL * 2*K / Ms
w_ex = GAMMALL * 2*A/Ms * k**2
w_c = GAMMALL * 4*A_c/(a**2 * Ms)
w_nn = GAMMALL * A_nn/Ms * k**2

# Magnon waves
w_mag = np.sqrt((w_ani + w_ex - w_nn)*(w_ani + w_ex - 2*w_c + w_nn))
omega_magn1 = w_mag + w_ext
omega_magn2 = w_mag - w_ext
omega_magn3 = -w_mag + w_ext
omega_magn4 = -w_mag - w_ext
ax.plot(k*1e-9, omega_magn1/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls, label="Model")
ax.plot(k*1e-9, omega_magn2/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)
ax.plot(k*1e-9, omega_magn3/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)
ax.plot(k*1e-9, omega_magn4/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)

# plot numerical result
ax.imshow(FT_tot, aspect='auto', origin='lower', extent=extent,
           vmin=0, vmax=1, cmap="inferno")

# plot cleanup
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Wavenumber (rad/nm)")
ax.set_ylabel("Frequency (THz)")
ax.legend(loc="lower right", fontsize="5")

# plot the zoomed in insert

# plotting ranges
ymin_zoom, ymax_zoom = 0.45, 0.6  # THz
factor = (ymax_zoom - ymin_zoom) / (ymax - ymin)
xmax_zoom = factor * (xmax - xmin) * 0.5
xmin_zoom = -xmax_zoom

ax2 = ax.inset_axes(bounds=[0.35, 0.65, 0.3, 0.3], xlim=[xmin_zoom, xmax_zoom],
                    ylim=[ymin_zoom, ymax_zoom])
linewidth = 0.5

# x- and y-coordinates of FT cell centers
ks = np.fft.fftshift(np.fft.fftfreq(nx_zoom, dx_zoom) * 2*np.pi)
fs = np.fft.fftshift(np.fft.fftfreq(nt_zoom, dt_zoom))
# FT cell widths
dk = ks[1] - ks[0]
df = fs[1] - fs[0]
# image extent of k-values and frequencies, compensated for cell-width
extent = [(ks[0] - dk/2) * 1e-9, (ks[-1] + dk/2) * 1e-9,
          (fs[0] - df/2) * 1e-12, (fs[-1] + df/2) * 1e-12]

# normalize them in the relevant area, so they are visible in the plot
x_start = int((xmin_zoom - extent[0]) / (extent[1] - extent[0]) * nx_zoom)
x_end = int((xmax_zoom - extent[0]) / (extent[1] - extent[0]) * nx_zoom)
y_start = int((ymin_zoom- extent[2]) / (extent[3] - extent[2]) * nt_zoom)
y_end = int((ymax_zoom - extent[2]) / (extent[3] - extent[2]) * nt_zoom)

# Fourier in time and x-direction of displacement and magnetization
m_FT = np.zeros((3, nt_zoom, nx_zoom))

for i in range(3):
    m_zoom_i = np.abs(np.fft.fftshift(np.fft.fft2(m_zoom[:,i,0,0,:])))
    m_zoom_max_i = np.max(m_zoom_i[y_start:y_end, x_start:x_end])
    m_zoom_i /= m_zoom_max_i
    m_FT[i,...] = m_zoom_i

FT_tot = m_FT[1,...]

# numerical calculations
k = np.linspace(xmin_zoom*1e9, xmax_zoom*1e9, 2000)

# spin waves
w_ext = GAMMALL * Bz
w_ani = GAMMALL * 2*K / Ms
w_ex = GAMMALL * 2*A/Ms * k**2
w_c = GAMMALL * 4*A_c/(a**2 * Ms)
w_nn = GAMMALL * A_nn/Ms * k**2

# Magnon waves
w_mag = np.sqrt((w_ani + w_ex - w_nn)*(w_ani + w_ex - 2*w_c + w_nn))
omega_magn1 = w_mag + w_ext
omega_magn2 = w_mag - w_ext
omega_magn3 = -w_mag + w_ext
omega_magn4 = -w_mag - w_ext
ax2.plot(k*1e-9, omega_magn1/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls, label="Model")
ax2.plot(k*1e-9, omega_magn2/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)
ax2.plot(k*1e-9, omega_magn3/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)
ax2.plot(k*1e-9, omega_magn4/(2*np.pi)*1e-12, color="green", lw=linewidth, ls=ls)

# plot numerical result
ax2.imshow(FT_tot, aspect='auto', origin='lower', extent=extent,
           vmin=0, vmax=1, cmap="inferno")

# plot cleanup
ax2.set_xlim(xmin_zoom, xmax_zoom)
ax2.set_ylim(ymin_zoom, ymax_zoom)

ax2.spines['bottom'].set_color('white')
ax2.spines['top'].set_color('white') 
ax2.spines['right'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.tick_params(axis='x', bottom=False, labelbottom=False)
ax2.tick_params(axis='y', left=False, labelleft=False)
ax2.yaxis.label.set_color('white')
ax2.xaxis.label.set_color('white')

# zoom lines
rect = [xmin_zoom, ymin_zoom, xmax_zoom-xmin_zoom, ymax_zoom-ymin_zoom]
box = ax.indicate_inset(rect, edgecolor="white", lw=0.2,
                        ls=":", alpha=1)
cp1 = ConnectionPatch(xyA=(xmin_zoom, ymax_zoom), xyB=(0, 0), axesA=ax, axesB=ax2,
                      coordsA="data", coordsB="axes fraction", lw=0.7, ls=":", color="white")
cp2 = ConnectionPatch(xyA=(xmax_zoom, ymax_zoom), xyB=(1, 0), axesA=ax, axesB=ax2,
                      coordsA="data", coordsB="axes fraction", lw=0.7, ls=":",
                      color="white")
cp1.set_zorder(11)
cp2.set_zorder(10)

ax.add_patch(cp1)
ax.add_patch(cp2)

plt.tight_layout()
plt.savefig("AFM_dispersion.pdf", dpi=1200)