"""
This script simulates and plots the data from Fig. 5 of the paper
'mumax+: extensible GPU-accelerated micromagnetics and beyond' (https://arxiv.org/abs/2411.18194)

The peak GPU memory usage can be logged using the `nvidia-smi` command.

Note that an installation of mumax3 is necessary in order to generate the full plot.
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import time

from mumaxplus import Ferromagnet, Grid, World

# Actual simulation. This is mostly identical to the sript in examples/bench.py
def simple_bench(grid, nsteps=100):
    """Returns the walltime of a simple simulation using the specified grid
    and numbers of steps"""

    world = World((4e-9, 4e-9, 4e-9))

    magnet = Ferromagnet(world, grid)
    magnet.enable_demag = True
    magnet.enable_openbc = True

    magnet.msat = 800e3
    magnet.aex = 13e-12
    magnet.alpha = 0.5

    world.timesolver.set_method("DormandPrince") # default in mumax3
    world.timesolver.timestep = 1e-13
    world.timesolver.adaptive_timestep = False

    world.timesolver.steps(10)  # warm up

    start = time.time()
    world.timesolver.steps(nsteps)
    stop = time.time()

    return stop - start


if __name__ == "__main__":
    NSTEPS = 100
    filename = "bench_mumaxplus.npy"
    ncells, throughput = [], []

    if not os.path.isfile(filename):

        for p in range(2, 12):
            grid = Grid((2 ** p, 2 ** p, 1))

            walltime = simple_bench(grid, NSTEPS)

            ncells.append(grid.ncells)
            throughput.append(grid.ncells * NSTEPS / walltime)

        np.save(filename, np.array([ncells, throughput]))
    else:
        ncells, throughput = np.load(filename)

    fig = plt.figure(figsize=(2.5, 4.8/6.4 * 2.5))

    if not os.path.isfile("bench_mumax3.out/benchmark.txt"):
        print("Benchmark data from mumax3 simulation not found. Please run `bench_mumax3.mx3`" \
        " using mumax3 in order to plot the full figure.")
    else:
        ncells_m3, TP_m3 = np.loadtxt("bench_mumax3.out/benchmark.txt", unpack=True)

        plt.loglog(ncells_m3, TP_m3, "-o", label=r"mumax$^3$ (32-bit)", markersize=5)
    # Default compilation is single precision
    plt.loglog(ncells, throughput,   "-^", label=r"mumax$^+$ (32-bit)", markersize=5)
    
    plt.legend(loc="lower right")
    plt.xlim(ncells[0], ncells[-1])
    plt.xlabel("Number of cells")
    plt.ylabel("Throughput (cells / s)")
    plt.tight_layout()
    plt.show()