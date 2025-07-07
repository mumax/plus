import numpy as np
from mumaxplus import Ferromagnet, Grid, World, _cpp
from mumaxplus.util import MU0

nx, ny, nz = 126, 64, 8
nx_aspect, ny_aspect, nz_aspect = 100, 100, 1
order, eps, R = 11, 5e-10, -1

def relative_error(result, wanted):
    return np.abs((wanted - result) / result)

def demag_field_py(magnet):
    kernel = _cpp._demag_kernel(magnet._impl, order, eps, R)
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
    h = -MU0 * np.array([hx, hy, hz]).real
    return h[
        :,
        (h.shape[1] - mag.shape[1]) :,
        (h.shape[2] - mag.shape[2]) :,
        (h.shape[3] - mag.shape[3]) :,
    ]

class TestDemag:
    def setup_class(self):
        """Readout all exact values from the .npy files and create the needed
           kernels. These kernels will be sliced in the tests in order to get
           the correct components and only 1/4th of the kernel that contains all
           information."""
        
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((nx, ny, nz)))
        self.kernel = _cpp._demag_kernel(magnet._impl, order, eps, R)

        # in the aspect tests, the cellsizes are different
        world = World((1e-9, 1.27e-9, 1.13e-9))
        magnet = Ferromagnet(world, Grid((nx_aspect, ny_aspect, nz_aspect)))
        self.kernel_aspect = _cpp._demag_kernel(magnet._impl, order, eps, R)

        # open all files
        self.exact_Nxx = np.load("exact_Nxx_3D.npy")

        self.exact_Nxy = np.load("exact_Nxy_3D.npy")[:,1:,1:]

        self.exact_aspect_Nxx = np.load("exact_Nxx_aspect.npy")

        self.exact_aspect_Nxy = np.load("exact_Nxy_aspect.npy")[:,1:,1:]

    def test_demagfield(self):
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((16, 4, 3)))
        wanted = demag_field_py(magnet)
        result = magnet.demag_field.eval()
        assert np.allclose(result, wanted)

    def test_Nxx_radius(self):
        """ Compare the demagkernel with high accurate .npy files. These were made
            with the BigFloat package with an accuracy of 1024 bits
            and the analytical method."""
        
        nx, ny, nz = 126, 64, 8
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((nx, ny, nz)))
        mumaxplus_result = _cpp._demag_kernel(magnet._impl, 11, 5e-10, 5e-9)[0,nz:,ny:, nx:] # Nxx component

        # avoid fake errors when both values are super small
        mask = ~((np.abs(self.exact_Nxx) < 5e-15) & (np.abs(mumaxplus_result) < 5e-15))

        rel_err = relative_error(mumaxplus_result, self.exact_Nxx)
        err = np.nanmax(rel_err[mask])

        assert err < 2e-4
    
    def test_Nxx(self):
        """ Compare the demagkernel with high accurate .npy files. These were made
            with the BigFloat package with an accuracy of 1024 bits
            and the analytical method."""
        
        mumaxplus_result = self.kernel[0,nz:,ny:, nx:] # Nxx component
        
        # avoid fake errors when both values are super small
        mask = ~((np.abs(self.exact_Nxx) < 5e-15) & (np.abs(mumaxplus_result) < 5e-15))

        rel_err = relative_error(mumaxplus_result, self.exact_Nxx)
        err = np.nanmax(rel_err[mask])
        assert err < 1e-4

    def test_Nxy(self):
        """ Compare the demagkernel with high accurate .npy files. These were made
            with the BigFloat package with an accuracy of 1024 bits
            and the analytical method."""
        
        mumaxplus_result = self.kernel[3,nz:,ny+1:, nx+1:] # Nxy component

        err = np.max(relative_error(mumaxplus_result, self.exact_Nxy))
        assert err < 1e-4

    def test_Nxx_aspect(self):
        """ Compare the demagkernel with high accurate .npy files. These were made
            with the BigFloat package with an accuracy of 1024 bits
            and the analytical method."""
        
        mumaxplus_result = self.kernel_aspect[0,:,ny_aspect:, nx_aspect:] # Nxx component

        # avoid fake errors when both values are super small
        mask = ~((np.abs(self.exact_aspect_Nxx) < 3e-12) & (np.abs(mumaxplus_result) < 3e-12))

        rel_err = relative_error(mumaxplus_result, self.exact_aspect_Nxx)
        err = np.nanmax(rel_err[mask])
        assert err < 1e-4

    def test_Nxy_aspect(self):
        """ Compare the demagkernel with high accurate .npy files. These were made
            with the BigFloat package with an accuracy of 1024 bits
            and the analytical method."""
            
        mumaxplus_result = self.kernel_aspect[3,:,ny_aspect+1:, nx_aspect+1:] # Nxy component

        err = np.max(relative_error(mumaxplus_result, self.exact_aspect_Nxy))
        assert err < 1e-5