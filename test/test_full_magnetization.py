import numpy as np
from mumaxplus import Antiferromagnet, NcAfm, World, Grid


def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

def test_full_magnetization_afm():
    nx, ny, nz = 31, 120, 60
    world = World(cellsize=(1e-9, 1e-9, 1e-9))
    magnet = Antiferromagnet(world, Grid((nx, ny, nz)))
    # magnet should initialize with random magnetization

    # add random saturation magnetizations
    magnet.sub1.msat = 800e3 * np.random.normal(loc=1, scale=0.05, size=(1, nz, ny, nx))
    magnet.sub2.msat = 750e3 * np.random.normal(loc=1, scale=0.05, size=(1, nz, ny, nx))

    wanted = (magnet.sub1.full_magnetization.eval() + \
              magnet.sub2.full_magnetization.eval())
    result = magnet.full_magnetization.eval()
    err = max_relative_error(result, wanted)
    assert err < 1e-5  # only error should be single versus double precision

def test_full_magnetization_ncafm():
    nx, ny, nz = 31, 120, 60
    world = World(cellsize=(1e-9, 1e-9, 1e-9))
    magnet = NcAfm(world, Grid((nx, ny, nz)))
    # magnet should initialize with random magnetization

    # add random saturation magnetizations
    magnet.sub1.msat = 800e3 * np.random.normal(loc=1, scale=0.05, size=(1, nz, ny, nx))
    magnet.sub2.msat = 750e3 * np.random.normal(loc=1, scale=0.05, size=(1, nz, ny, nx))
    magnet.sub3.msat = 700e3 * np.random.normal(loc=1, scale=0.05, size=(1, nz, ny, nx))


    wanted = (magnet.sub1.full_magnetization.eval() + \
              magnet.sub2.full_magnetization.eval() + \
              magnet.sub3.full_magnetization.eval())
    result = magnet.full_magnetization.eval()
    err = max_relative_error(result, wanted)
    assert err < 1e-5  # only error should be single versus double precision
