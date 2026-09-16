import numpy as np
import pytest

from mumaxplus import Ferromagnet, Grid, World

cs = (1, 1, 1)  # cellsize

VALID_PBC = [
    ((0, 0, 0), Grid((0, 0, 0), (0, 0, 0))),
    ((2, 0, 0), Grid((128, 0, 0), (1, 2, 3))),
    ((0, 5, 0), Grid((0, 109, 0), (8, 0, 6))),
    ((0, 0, 3), Grid((0, 0, 32), (0, 6, 0))),
    ((2, 3, 4), Grid((128, 32, 64), (1, 0, 0))),
]

INVALID_PBC = [
    ((-1, 0, 0), Grid((20, 0, 0))),
    ((0, -1, 0), Grid((0, 16, 0))),
    ((0, 0, -1), Grid((0, 0, 8))),
    ((0.5, 0.6, 0.7), Grid((1, 1, 1))),
    ((2, 1), Grid((1, 1, 1))),
    ((0, 0, 0), Grid((128, 0, 0))),
    ((0, 0, 0), Grid((0, 21, 0))),
    ((0, 0, 0), Grid((0, 0, 64))),
    ((1, 0, 0), Grid((0, 0, 0))),
    ((0, 2, 0), Grid((0, 0, 0))),
    ((0, 0, 3), Grid((0, 0, 0))),
    ((2, 0, 3), Grid((0, 128, 0))),
]


def test_default_world_init():
    world = World(cs)
    assert world.pbc_repetitions == (0, 0, 0)
    assert world.mastergrid == Grid((0, 0, 0))


@pytest.mark.parametrize("pbc_repetitions,mastergrid", VALID_PBC)
def test_world_init(pbc_repetitions, mastergrid):
    world = World(cs, pbc_repetitions, mastergrid)
    assert world.pbc_repetitions == pbc_repetitions
    assert world.mastergrid == mastergrid


@pytest.mark.parametrize("pbc_repetitions,mastergrid", INVALID_PBC)
def test_invalid_world_init(pbc_repetitions, mastergrid):
    with pytest.raises((ValueError, TypeError)):
        world = World(cs, pbc_repetitions, mastergrid)


@pytest.mark.parametrize("pbc_repetitions,mastergrid", VALID_PBC)
def test_set_pbc(pbc_repetitions, mastergrid):
    world = World(cs)
    world.set_pbc(pbc_repetitions, mastergrid)
    assert world.pbc_repetitions == pbc_repetitions
    assert world.mastergrid == mastergrid


@pytest.mark.parametrize("pbc_repetitions,mastergrid", INVALID_PBC)
def test_invalid_set_pbc(pbc_repetitions, mastergrid):
    world = World(cs)
    with pytest.raises((ValueError, TypeError)):
        world.set_pbc(pbc_repetitions, mastergrid)


def bounding_grid(grids):
    """Get bounding grid"""
    minimum = grids[0].origin
    maximum = np.asarray(grids[0].origin) + np.asarray(grids[0].size)

    for grid in grids[1::]:
        minimum = np.minimum(minimum, grid.origin)
        maximum = np.maximum(maximum, np.asarray(grid.origin) + np.asarray(grid.size))

    return Grid(np.asarray(maximum) - np.asarray(minimum), origin=minimum)


def bounding_mastergrid(grids, pbc_repetitions):
    """Get bounding mastergrid"""
    bg = bounding_grid(grids)
    size = [bg.size[i] if pbc_repetitions[i] != 0 else 0 for i in range(3)]
    return Grid(size, origin=bg.origin)


@pytest.mark.parametrize(
    "grid1,grid2",
    [
        (Grid((4, 2, 1), (0, 0, 0)), Grid((1, 2, 3), (0, 15, 29))),
        (Grid((2, 4, 3), (-5, 7, 12)), Grid((8, 3, 4), (0, 1, 0))),
        (Grid((8, 1, 6), (1, 2, 3)), Grid((7, 2, 3), (-5, 6, -9))),
    ],
)
@pytest.mark.parametrize(
    "pbc_repetitions", [(1, 2, 1), (0, 1, 2), (2, 0, 1), (1, 2, 0), (0, 0, 0)]
)
def test_set_pbc_after_magnets(grid1, grid2, pbc_repetitions):
    world = World(cs)
    magnet = Ferromagnet(world, grid1)
    magnet = Ferromagnet(world, grid2)
    world.set_pbc(pbc_repetitions)
    assert world.pbc_repetitions == pbc_repetitions
    assert world.mastergrid == bounding_mastergrid([grid1, grid2], pbc_repetitions)


def test_set_pbc_no_master_before_magnet():
    world = World(cs)
    with pytest.raises(IndexError):
        world.set_pbc((1, 4, 3))


@pytest.mark.parametrize("pbc_repetitions,mastergrid", INVALID_PBC)
def test_invalid_set_pbc(pbc_repetitions, mastergrid):
    world = World(cs)
    with pytest.raises((ValueError, TypeError)):
        world.set_pbc(pbc_repetitions, mastergrid)


# could parametrize but whatever
def test_invalid_set_pbc_after_magnet():
    world = World(cs, pbc_repetitions=(1, 1, 1), mastergrid=Grid((2, 4, 6)))
    magnet = Ferromagnet(world, Grid((2, 4, 6)))
    with pytest.raises(IndexError):
        world.set_pbc(pbc_repetitions=(1, 2, 3), mastergrid=Grid((1, 2, 3)))


@pytest.mark.parametrize(
    "origin", [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (10, 0, 0), (0, 10, 0), (0, 0, 10)]
)
def test_magnet_not_in_mastergrid(origin):
    world = World(cs, pbc_repetitions=(1, 1, 1), mastergrid=Grid((2, 4, 6)))
    with pytest.raises(IndexError):
        magnet = Ferromagnet(world, Grid((1, 2, 3), origin=(-1, 0, 0)))


@pytest.mark.parametrize(
    "grid1,grid2",
    [
        (Grid((4, 2, 1), (0, 0, 0)), Grid((1, 2, 3), (0, 15, 29))),
        (Grid((2, 4, 3), (-5, 7, 12)), Grid((8, 3, 4), (0, 1, 0))),
        (Grid((8, 1, 6), (1, 2, 3)), Grid((7, 2, 3), (-5, 6, -9))),
    ],
)
def test_bounding_grid(grid1, grid2):
    world = World(cs)
    magnet = Ferromagnet(world, grid1)
    magnet = Ferromagnet(world, grid2)
    assert world.bounding_grid == bounding_grid([grid1, grid2])


def test_invalid_bounding_grid():
    world = World(cs)
    with pytest.raises(IndexError):
        bg = world.bounding_grid


# could parametrize but I think you get the point
def test_set_mastergrid():
    world = World(cs, pbc_repetitions=(0, 1, 2), mastergrid=Grid((0, 12, 16)))
    new_mastergrid = Grid((0, 1, 6))
    world.mastergrid = new_mastergrid
    assert world.mastergrid == new_mastergrid


# could parametrize but I think you get the point
def test_set_invalid_mastergrid():
    world = World(cs, pbc_repetitions=(0, 1, 2), mastergrid=Grid((0, 12, 16)))
    new_mastergrid = Grid((1, 0, 6))
    with pytest.raises(ValueError):
        world.mastergrid = new_mastergrid


def test_set_invalid_mastergrid_after_magnet():
    world = World(cs, pbc_repetitions=(0, 1, 2), mastergrid=Grid((0, 6, 2)))
    magnet = Ferromagnet(world, Grid((4, 6, 2)))
    new_mastergrid = Grid((0, 3, 1))
    with pytest.raises(IndexError):
        world.mastergrid = new_mastergrid


def test_set_pbc_repetitions():
    world = World(cs, pbc_repetitions=(1, 2, 3), mastergrid=Grid((5, 4, 3)))
    world.pbc_repetitions = (3, 2, 1)
    assert world.pbc_repetitions == (3, 2, 1)


@pytest.mark.parametrize(
    "new_rep", [(0, 1, 2), (1, 0, 2), (1, 2, 0), (-1, 2, 3), (4, 0.5, 6)]
)
def test_set_invalid_pbc_repetitions(new_rep):
    world = World(cs, pbc_repetitions=(1, 2, 3), mastergrid=Grid((5, 4, 3)))
    with pytest.raises((ValueError, TypeError)):
        world.pbc_repetitions = new_rep


def test_unset_pbc():
    world = World(
        cs,
        pbc_repetitions=(1, 2, 3),
        mastergrid=Grid((128, 32, 26), origin=(-12, 23, 8)),
    )
    world.unset_pbc()
    assert world.pbc_repetitions == (0, 0, 0)
    assert world.mastergrid == Grid((0, 0, 0))
