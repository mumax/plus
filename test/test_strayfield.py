import numpy as np
import pytest

from mumax5 import Ferromagnet, Grid, StrayField, World


def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)


def extended_grid(grid1, grid2):
    """ Returns a grid which contains the two grids """

    bottom_left = [min(grid1.origin[c], grid2.origin[c]) for c in [0, 1, 2]]

    top_right = [
        max(grid1.origin[c] + grid1.size[c], grid2.origin[c] + grid2.size[c])
        for c in [0, 1, 2]
    ]

    origin = bottom_left
    size = [top_right[c] - bottom_left[c] for c in [0, 1, 2]]

    return Grid(size, origin)


TESTCASES = [
    {
        "mgrid": Grid(size=(17, 4, 1), origin=(3, 6, 0)),
        "hgrid": Grid(size=(17, 4, 1), origin=(3, 6, 0)),
    },
    {
        "mgrid": Grid(size=(8, 10, 1), origin=(2, 15, 0)),
        "hgrid": Grid(size=(6, 7, 2), origin=(-5, 0, 2)),
    },
    {
        "mgrid": Grid(size=(8, 10, 1), origin=(2, 15, 0)),
        "hgrid": Grid(size=(32, 32, 1), origin=(-5, 0, 0)),
    },
    {
        "mgrid": Grid(size=(8, 10, 1), origin=(2, 15, -1)),
        "hgrid": Grid(size=(32, 32, 1), origin=(-5, 0, 0)),
    },
    {
        "mgrid": Grid(size=(8, 10, 2), origin=(2, 15, -1)),
        "hgrid": Grid(size=(32, 32, 1), origin=(-5, 0, 1)),
    },
    {
        "mgrid": Grid(size=(8, 10, 2), origin=(2, 15, 2)),
        "hgrid": Grid(size=(32, 32, 5), origin=(0, 0, 1)),
    },
]


class TestStrayFields:
    @pytest.mark.parametrize("test_case", TESTCASES)
    def test_fft_against_brute(self, test_case):
        """Computes the H-field in hgrid of a magnet on mgrid using both the
        fft method and the brute method. Asserts that both methods yield the
        same results.
        """

        mgrid, hgrid = test_case["mgrid"], test_case["hgrid"]

        world = World((1e-9, 2e-9, 3.1e-9))
        magnet = Ferromagnet(world, mgrid)
        magnet.msat = 800e3
        magnet.magnetization = (1, 0, 0)

        mf = StrayField(magnet, hgrid)
        mf.set_method("fft")
        result = mf.eval()
        mf.set_method("brute")
        wanted = mf.eval()

        assert max_relative_error(result, wanted) < 1e-4

    @pytest.mark.parametrize("test_case", TESTCASES)
    def test_magnetfields(self, test_case):
        """Asserts that the H-field in hgrid of a magnet on mgrid is computed
        correctly.

        This is done by comparing the result with the demag field of a magnet on
        a grid which containds both the mgrid and the hgrid. The magnetization
        is set to zero, except inside the mgrid in order to create an equivalent
        system as the system which is being checked.
        """

        mgrid, hgrid = test_case["mgrid"], test_case["hgrid"]

        world = World((1e-9, 2e-9, 3.1e-9))
        magnet = Ferromagnet(world, mgrid)
        magnet.msat = 800e3
        magnet.magnetization = (1, -1, 2)

        # Compute the hfield in hgrid of the magnet in mgrid.
        # This is the field which which need to be checked.
        hfield = StrayField(magnet, hgrid).eval()

        # Construct an equivalent system against we will check the result.
        world2 = World(world.cellsize)
        box = extended_grid(mgrid, hgrid)
        magnet2 = Ferromagnet(world2, box)

        mxi, myi, mzi = [mgrid.origin[c] - box.origin[c] for c in [0, 1, 2]]
        mxf, myf, mzf = [
            mgrid.origin[c] + mgrid.size[c] - box.origin[c] for c in [0, 1, 2]
        ]

        hxi, hyi, hzi = [hgrid.origin[c] - box.origin[c] for c in [0, 1, 2]]
        hxf, hyf, hzf = [
            hgrid.origin[c] + hgrid.size[c] - box.origin[c] for c in [0, 1, 2]
        ]

        ms = np.zeros(magnet2.msat.eval().shape)
        ms[:, mzi:mzf, myi:myf, mxi:mxf] = magnet.msat.eval()
        magnet2.msat = ms

        m = np.ones(magnet2.magnetization.eval().shape)
        m[:, mzi:mzf, myi:myf, mxi:mxf] = magnet.magnetization.eval()
        magnet2.magnetization = m

        wanted = magnet2.demag_field.eval()[:, hzi:hzf, hyi:hyf, hxi:hxf]

        assert max_relative_error(hfield, wanted) < 1e-4
