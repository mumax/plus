from mumaxplus import World, Grid, Antiferromagnet, Ferromagnet


def test_is_not_sublattice():
    world = World(cellsize=(4e-9, 5e-9, 6e-9))
    magnet = Ferromagnet(world, Grid((16, 8, 2)))
    assert not magnet.is_sublattice

def test_is_sublattice():
    world = World(cellsize=(3e-9, 2e-9, 1e-9))
    magnet = Antiferromagnet(world, Grid((8, 16, 2)))
    assert magnet.sub1.is_sublattice and magnet.sub2.is_sublattice
