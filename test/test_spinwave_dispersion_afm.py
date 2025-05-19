"""Antiferromagnetic spinwave dispersion relation
This test copies the example and automatically tests the difference between the
analytical and simulation results within the valid range.
"""

import numpy as np

from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util.constants import GAMMALL

RTOL = 3e-2  # 3%


def test_spinwave_dispersion_afm():
    # Numerical parameters
    fmax = 5E13          # maximum frequency (in Hz) of the sinc pulse
    T = 2E-12            # simulation time (longer -> better frequency resolution)
    dt = 1 / (2 * fmax)  # the sample time
    dx = 1E-9            # cellsize
    nx = 512             # number of cells

    # Material/system parameters
    Bz = 0.2           # bias field along the z direction
    A = 10E-12         # exchange constant
    A_nn = -5E-12
    A_c = -400E-12
    Ms = 400e3         # saturation magnetization
    alpha = 0.005      # damping parameter
    gamma = GAMMALL    # gyromagnetic ratio
    K = 1e3
    mu0 = 1.256637062E-6

    # Create the world
    grid_size = (nx, 1, 1)
    cell_size = (dx, dx, dx)

    world = World(cell_size)
    world.bias_magnetic_field = (0, 0, Bz)

    # Create a ferromagnet
    magnet = Antiferromagnet(world, Grid(size=grid_size))
    for sub in magnet.sublattices:
        sub.msat = Ms
        sub.aex = A
        sub.alpha = alpha
        sub.ku1 = K
        sub.anisU = (0, 0, 1)

    magnet.afmex_nn = A_nn
    magnet.afmex_cell = A_c
    magnet.latcon = dx


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
    mx = m[:, 0, 0, 0, :]  # time, xcomp, z, y, x-axis
    mx_fft = np.fft.fft2(mx)
    mx_fft = np.fft.fftshift(mx_fft)

    # Find maximum amplitude frequencies of FFT
    real_fft = np.abs(mx_fft)**2  # to real
    positive_fft = real_fft[real_fft.shape[0]//2:, :]  # keep positive part
    freq_mumaxplus = 1/T * np.argmax(positive_fft, axis=0)  # maximum of each column

    # The analytically derived dispersion relation
    k = np.linspace(-np.pi/dx, np.pi/dx * (nx-2)/nx, nx)
    mu0Ms = mu0 * Ms
    He = -4 * A_c / (dx*dx*mu0Ms)
    Ha = 2 * K / mu0Ms
    Hint = (2*A - A_nn) / mu0Ms
    freq_theory = mu0 * gamma / (2*np.pi) * np.sqrt((2*He + Ha + Hint * k**2) * (Ha + Hint * k**2)) + Bz * gamma/(2*np.pi)

    # difference
    freq_diff = abs(freq_mumaxplus - freq_theory)
    valid_k_i = np.argmin(abs(k - (-1/dx)))
    valid_k_j = np.argmin(abs(k - 1/dx))
    valid_freq_diff = freq_diff[valid_k_i:valid_k_j]  # only keep valid part

    semi_relative_error = max(valid_freq_diff/fmax)

    assert semi_relative_error < RTOL
