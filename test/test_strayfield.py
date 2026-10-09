import numpy as np
import pytest

from mumaxplus import Ferromagnet, Grid, StrayField, World


def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)


def extended_grid(grid1, grid2):
    """Returns a grid which contains the two grids"""

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
    @pytest.mark.parametrize("pbc_repetitions", [None, (1, 2, 0)])
    @pytest.mark.parametrize("test_case", TESTCASES)
    def test_fft_against_brute(self, test_case, pbc_repetitions):
        """Computes the H-field in hgrid of a magnet on mgrid using both the
        fft method and the brute method. Asserts that both methods yield the
        same results.
        """

        mgrid, hgrid = test_case["mgrid"], test_case["hgrid"]

        world = World((1e-9, 2e-9, 3.1e-9))
        magnet = Ferromagnet(world, mgrid)
        magnet.msat = 800e3
        magnet.magnetization = (1, 0, 0)

        if pbc_repetitions is not None:
            world.set_pbc(pbc_repetitions)

        mf = StrayField(magnet, hgrid)
        mf.set_method("fft")
        result = mf.eval()
        mf.set_method("brute")
        wanted = mf.eval()

        assert max_relative_error(result, wanted) < 1e-4

    @pytest.mark.parametrize("pbc_repetitions", [None, (1, 2, 1)])
    @pytest.mark.parametrize("test_case", TESTCASES)
    def test_magnetfields(self, test_case, pbc_repetitions):
        """Asserts that the H-field in hgrid of a magnet on mgrid is computed
        correctly.

        This is done by comparing the result with the demag field of a magnet on
        a grid which contains both the mgrid and the hgrid. The magnetization
        is set to zero, except inside the mgrid in order to create an equivalent
        system as the system which is being checked.
        """

        mgrid, hgrid = test_case["mgrid"], test_case["hgrid"]
        box = extended_grid(mgrid, hgrid)

        world = World(
            (1e-9, 2e-9, 3.1e-9),
            (0, 0, 0) if pbc_repetitions is None else pbc_repetitions,
            Grid((0, 0, 0)) if pbc_repetitions is None else box,
        )
        magnet = Ferromagnet(world, mgrid)
        magnet.msat = 800e3
        magnet.magnetization = (1, -1, 2)

        # Compute the hfield in hgrid of the magnet in mgrid.
        # This is the field which which need to be checked.
        hfield = StrayField(magnet, hgrid).eval()

        # Construct an equivalent system against we will check the result.
        world2 = World(
            world.cellsize,
            (0, 0, 0) if pbc_repetitions is None else pbc_repetitions,
            Grid((0, 0, 0)) if pbc_repetitions is None else box,
        )
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


# --------------------------------------------------


@pytest.fixture(scope="class", params=[None, (1, 1, 0), (0, 0, 1)])
def multi_magnets(request):
    pbc = request.param

    world = World((1e-9, 2e-9, 3.1e-9))

    n_min, n_max = 2, 16
    offset = np.array((n_max, n_max, n_max))

    n_magnets = 4
    magnets = []
    for i in range(n_magnets):
        grid = Grid(
            size=tuple(np.random.randint(n_min, n_max, (3))), origin=tuple(i * offset)
        )
        magnet = Ferromagnet(world, grid)
        magnet.aex = 13e-12
        magnet.msat = 800e3
        magnets.append(magnet)

    if pbc is not None:
        world.set_pbc(pbc_repetitions=pbc)

    return world, magnets


class TestEnableStrayFields:
    """Test enable_strayfield_as_source/destination"""

    def test_enable_as_stray_field_destination(self, multi_magnets):
        w, magnets = multi_magnets

        for magnet in magnets:
            magnet.enable_as_stray_field_destination = False
            assert np.all(abs(magnet.external_field.eval()) < 1e-15)
            magnet.enable_as_stray_field_destination = True

    def test_enable_as_stray_field_source(self, multi_magnets):
        w, magnets = multi_magnets
        n_magnets = len(magnets)

        # evaluate all possible external strayfields
        strayfields_src_dst = {}
        for src in magnets:
            strayfields_src_dst[src.name] = {}
            for dst in magnets:
                if src.name == dst.name:  # skip demag
                    continue
                sf = StrayField(src, dst.grid)
                strayfields_src_dst[src.name][dst.name] = sf.eval()

        for i in range(2**n_magnets):
            # go through all True/False permutations
            permutation = [bool(int(j)) for j in bin(i)[2:].zfill(n_magnets)]
            for enable, magnet in zip(permutation, magnets):
                magnet.enable_as_stray_field_source = enable

            for dst in magnets:  # check every magnet as a destination
                result = dst.external_field.eval()

                wanted = np.zeros_like(result)
                for enable, src in zip(permutation, magnets):
                    if (src.name == dst.name) or not enable:
                        continue
                    wanted += strayfields_src_dst[src.name][dst.name]

                assert np.all(abs(result - wanted) < 1e-10)
