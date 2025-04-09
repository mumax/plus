import numpy as np
import json
from mumaxplus import Ferromagnet, Grid, World, _cpp
import matplotlib.pyplot as plt

def demag_field_py(magnet):
    kernel = _cpp._demag_kernel(magnet._impl)
    mag = magnet.msat.average() * magnet.magnetization.get()
    # add padding to the magnetization so that the size of magnetization
    # matches the size of the kernel
    pad = (
        (0, kernel.shape[1] - mag.shape[1]),
        (0, kernel.shape[2] - mag.shape[2]),
        (0, kernel.shape[3] - mag.shape[3]),
    )
    m = np.pad(mag, ((0, 0), *pad), "constant")

    # fourier transform of the magnetization and the kernel
    m = np.fft.fftn(m, axes=(1, 2, 3))
    kxx, kyy, kzz, kxy, kxz, kyz = np.fft.fftn(kernel, axes=(1, 2, 3))

    # apply the kernel and perform inverse fft
    hx = np.fft.ifftn(m[0] * kxx + m[1] * kxy + m[2] * kxz)
    hy = np.fft.ifftn(m[0] * kxy + m[1] * kyy + m[2] * kyz)
    hz = np.fft.ifftn(m[0] * kxz + m[1] * kyz + m[2] * kzz)

    # return the real part
    mu0 = 4 * np.pi * 1e-7
    h = -mu0 * np.array([hx, hy, hz]).real
    return h[
        :,
        (h.shape[1] - mag.shape[1]) :,
        (h.shape[2] - mag.shape[2]) :,
        (h.shape[3] - mag.shape[3]) :,
    ]


class TestDemag:
    def test_demagfield(self):
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((16, 4, 3)))
        wanted = demag_field_py(magnet)
        result = magnet.demag_field.eval()
        err = np.max(np.abs((wanted - result) / result))
        assert err < 2e-3

    # Compare the demagkernel with high accurate json files
    def test_Nxx(self):
        nx, ny, nz = 100, 100, 1
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((nx, ny, nz)))
        mumaxplus = _cpp._demag_kernel(magnet._impl)[0,:,ny:, nx:] # Nxx component
        
        f = open("exact_Nxx", "r")
        exact = np.array(json.loads(f.read()), dtype=float)

        err = np.max(np.abs((exact - mumaxplus) / exact))
        assert err < 1e-4

    def test_Nxy(self):
        nx, ny, nz = 100, 100, 1
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((nx, ny, nz)))
        mumaxplus = _cpp._demag_kernel(magnet._impl)[3,:,ny+1:, nx+1:] # Nxy component

        f = open("exact_Nxy", "r")
        exact = np.array(json.loads(f.read()), dtype=float)[:,1:,1:]

        err = np.max(np.abs((exact - mumaxplus) / exact))
        assert err < 1e-5

    def test_Nxx_aspect(self):
        nx, ny, nz = 100, 100, 1
        world = World((1e-9, 1.27e-9, 1.13e-9))
        magnet = Ferromagnet(world, Grid((nx, ny, nz)))
        mumaxplus = _cpp._demag_kernel(magnet._impl)[0,:,ny:, nx:] # Nxx component

        f = open("exact_Nxx_aspect", "r")
        exact = np.array(json.loads(f.read()), dtype=float)

        err = np.max(np.abs((exact - mumaxplus) / exact))
        assert err < 1e-2

    def test_Nxy_aspect(self):
        nx, ny, nz = 100, 100, 1
        world = World((1e-9, 1.27e-9, 1.13e-9))
        magnet = Ferromagnet(world, Grid((nx, ny, nz)))
        mumaxplus = _cpp._demag_kernel(magnet._impl)[3,:,ny+1:, nx+1:] # Nxy component

        f = open("exact_Nxy_aspect", "r")
        exact = np.array(json.loads(f.read()), dtype=float)[:,1:,1:]
        
        err = np.max(np.abs((exact - mumaxplus) / exact))
        assert err < 1e-5