"""Altermagnetic spinwave dispersion relation.
"""

import numpy as np
import pytest

from mumaxplus import Altermagnet, Grid, World
from mumaxplus.util.constants import GAMMALL

# Value of RTOL is a trade-off between necessary simulation time and accuracy.
RTOL = 2e-2  # 2%

@pytest.mark.slow
def test_spinwave_dispersion_afm():
    # Numerical parameters
    fmax = 1E13           # maximum frequency (in Hz) of the sinc pulse
    T = 20E-12            # simulation time (longer -> better frequency resolution)
    dt = 1 / (2 * fmax)   # the sample time
    dx = 0.5E-9           # cellsize
    nx = 2048             # number of cells

    # Material/system parameters
    Bz = 0.1
    A1 = 25e-12
    A2 = 15e-12
    A0 = -5e-13
    A12 = A0/2
    angle = 0
    Ms = 2.9e5
    alpha = 0.001
    K = 2e5

    # Create the world
    grid_size = (nx, 1, 1)
    cell_size = (dx, dx, dx)

    world = World(cell_size)
    world.bias_magnetic_field = (0, 0, Bz)

    magnet = Altermagnet(world, Grid(size=grid_size))

    magnet.msat = Ms
    magnet.alpha = alpha
    magnet.latcon = dx
    magnet.ku1 = K
    magnet.anisU = (0, 0, 1)

    magnet.atmex_nn = A12
    magnet.atmex_cell = A0
    magnet.A1 = A1
    magnet.A2 = A2
    magnet.angle = angle

    magnet.enable_demag = False

    Bt = lambda t: (1e2* np.sinc(2 * fmax * (t - T / 2)), 0, 0)
    mask = np.zeros(shape=(1, 1, nx))
    # Put signal at the center of the simulation box
    mask[:, :, nx // 2 - 1:nx // 2 + 1] = 1
    for sub in magnet.sublattices:
        sub.bias_magnetic_field.add_time_term(Bt, mask)

    magnet.sub1.magnetization = (0, 0, 1)
    magnet.sub2.magnetization = (0, 0, -1)
    magnet.minimize()

    nt = 1 + int(T / dt)
    timepoints = np.linspace(0, T, nt)
    outputquantities = {'m': lambda: magnet.sub1.magnetization.eval()}

    # Run solver
    output = world.timesolver.solve(timepoints, outputquantities)

    # Apply the two dimensional FFT
    m = np.array(output['m'], dtype=float)
    my = m[:, 1, 0, 0, :]  # time, ycomp, z, y, x-axis
    my_fft = np.fft.fft2(my)
    my_fft = np.fft.fftshift(my_fft)

    # Find maximum amplitude frequencies of FFT
    real_fft = np.abs(my_fft)**2  # to real
    positive_fft = real_fft[real_fft.shape[0]//2:, :]  # keep positive part
    freq_mumaxplus = 1/T * np.argmax(positive_fft, axis=0)  # maximum of each column

    # The analytically derived dispersion relation
    k = np.linspace(-np.pi/dx, np.pi/dx * (nx-2)/nx, nx)
    kx = k * np.cos(angle)
    ky = k * np.sin(angle)

    wext = Bz
    wani = 2 * K / Ms
    wc = 4*A0/(dx*dx*Ms)
    wnn = A12/Ms * k**2

    wex = 0.5*(A1+A2) * k**2 / Ms
    walt = 0.5*(A1-A2) * (np.cos(2*angle)*(kx**2 - ky**2) + 2*np.sin(2*angle)*kx*ky) / Ms

    wmagnon = np.sqrt((wani + wex - wnn) * (wani + wex - 2*wc + wnn))
    freq_theory = GAMMALL * (wmagnon - wext + walt) / (2 * np.pi)

    # difference
    freq_diff = abs(freq_mumaxplus - freq_theory)

    # only consider middle 20%
    valid_freq_diff = freq_diff[int(nx * 0.4):int(nx * 0.6)]

    semi_relative_error = max(valid_freq_diff/fmax)
    assert semi_relative_error < RTOL