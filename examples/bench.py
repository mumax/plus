import time

import matplotlib.pyplot as plt

from mumaxplus import Ferromagnet, Grid, World


def simple_bench(grid, nsteps=100):
    """Returns the walltime of a simple simulation using the specified grid
    and numbers of steps"""

    world = World((4e-9, 4e-9, 4e-9))

    magnet = Ferromagnet(world, grid)
    magnet.msat = 800e3
    magnet.aex = 13e-12
    magnet.alpha = 0.5

    world.timesolver.timestep = 1e-13
    world.timesolver.adaptive_timestep = False

    world.timesolver.steps(10)  # warm up

    start = time.time()
    world.timesolver.steps(nsteps)
    stop = time.time()

    return stop - start


if __name__ == "__main__":
    NSTEPS = 100

    ncells = []
    throughputs = []

    print("\n{:>10} {:>10} {:>12}".format("ncells", "walltime", "throughput"))

    for p in range(2, 12):
        grid = Grid((2 ** p, 2 ** p, 1))
        walltime = simple_bench(grid, NSTEPS)
        throughput = grid.ncells * NSTEPS / walltime

        print("{:>10} {:>10.5f} {:>12.3E}".format(grid.ncells, walltime, throughput))

        ncells.append(grid.ncells)
        throughputs.append(throughput)

    print()

    plt.loglog(ncells, throughputs, "-o")
    plt.xlabel("Number of cells")
    plt.ylabel("Throughput")
    plt.show()
