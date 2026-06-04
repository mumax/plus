import numpy as np
from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.shape import Ellipsoid


C11, C12, C44 = 300e-9, 110e-9, 60e-9  # not explicitely used, but avoids assuredZero checks
cx, cy, cz = 1e-9, 2e-9, 3e-9  # non-equal cell sizes

nx, ny, nz = 32, 16, 8
max_traction = 1e9  # 1 GPa

class TestBoundaryTraction3DGeom:
    """Test the inclusion of spatially resolved boundary traction into the
    internal force calculation with non-trivial geometry. No displacement is
    set, so there is no stress. This isolates the effect of the applied boundary
    traction.
    """

    def setup_class(self):
        geom = Ellipsoid(nx*cx, ny*cy, nz*cz)
        geom -= Ellipsoid(nx*cx/2, ny*cy/2, nz*cz/2)  # make hole
        geom.translate((nx - 1)/2 * cx, (ny - 1)/2 * cy, (nz - 1)/2 * cz)  # to center

        self.world = World((cx, cy, cz))
        self.magnet = Ferromagnet(self.world, Grid((nx, ny, nz)), geometry=geom)
        self.magnet.enable_elastodynamics = True
        self.magnet.C11, self.magnet.C12, self.magnet.C44 = C11, C12, C44

        # random traction to set
        self.traction = np.random.uniform(low=-max_traction, high=max_traction, size=(3, nz, ny, nx))

    def get_force(self, orientation, sense):
        """Get numerical force calculation for a specific orientation
        (0: x, 1: y, 2: z) and sense (-1: neg, +1: pos).

        This automatically finds cells with appropriate boundaries and calculates
        term including traction of the finite difference stencil of the internal
        force.
        """
        geom = self.magnet.geometry
        geom_shifted = np.roll(geom, -sense, axis=[2-orientation])

        # new geometry side is empty space: 0
        index = [slice(None)]*3
        # slice the appropriate side of the appropriate axis
        index[2-orientation] = 0 if sense == -1 else -1
        geom_shifted[tuple(index)] = 0  # no geom

        # boundary where there used to be geometry, but not after shifting
        boundaries = np.logical_and(geom, np.logical_not(geom_shifted))

        num_force = np.zeros((3, nz, ny, nx))
        for c in range(3):
            # 4/3 from stencil; 1/cellsize from derivative
            num_force[c,...] = 1 / self.world.cellsize[orientation] * 4./3. * boundaries * self.traction[c,...]
        
        return num_force


    def test_neg_x_side(self):
        self.magnet.boundary_traction.make_zero()
        self.magnet.boundary_traction.neg_x_side = self.traction

        mxp_force = self.magnet.internal_body_force.eval()
        num_force = self.get_force(0, -1)

        assert np.allclose(mxp_force, num_force, atol=1e-7 * max_traction / cx)

    def test_pos_x_side(self):
        self.magnet.boundary_traction.make_zero()
        self.magnet.boundary_traction.pos_x_side = self.traction

        mxp_force = self.magnet.internal_body_force.eval()
        num_force = self.get_force(0, 1)

        assert np.allclose(mxp_force, num_force, atol=1e-7 * max_traction / cx)

    def test_neg_y_side(self):
        self.magnet.boundary_traction.make_zero()
        self.magnet.boundary_traction.neg_y_side = self.traction

        mxp_force = self.magnet.internal_body_force.eval()
        num_force = self.get_force(1, -1)

        assert np.allclose(mxp_force, num_force, atol=1e-7 * max_traction / cy)

    def test_pos_y_side(self):
        self.magnet.boundary_traction.make_zero()
        self.magnet.boundary_traction.pos_y_side = self.traction

        mxp_force = self.magnet.internal_body_force.eval()
        num_force = self.get_force(1, 1)

        assert np.allclose(mxp_force, num_force, atol=1e-7 * max_traction / cy)

    def test_neg_z_side(self):
        self.magnet.boundary_traction.make_zero()
        self.magnet.boundary_traction.neg_z_side = self.traction

        mxp_force = self.magnet.internal_body_force.eval()
        num_force = self.get_force(2, -1)

        assert np.allclose(mxp_force, num_force, atol=1e-7 * max_traction / cz)

    def test_pos_z_side(self):
        self.magnet.boundary_traction.make_zero()
        self.magnet.boundary_traction.pos_z_side = self.traction

        mxp_force = self.magnet.internal_body_force.eval()
        num_force = self.get_force(2, 1)

        assert np.allclose(mxp_force, num_force, atol=1e-7 * max_traction / cz)


# ==================================================

# some different values of traction
neg_side_traction = (0.11e9, -0.21e9, 0.14e9)
pos_side_traction = (-0.05e9, 0.13e9, -0.32e9)

def get_2D_magnet(orientation):
    """Make small flat elastic magnet."""
    grid_size = [3, 3, 3]
    grid_size[orientation] = 1  # make flat

    world = World((cx, cy, cz))
    magnet = Ferromagnet(world, Grid(grid_size))
    magnet.enable_elastodynamics = True
    magnet.C11, magnet.C12, magnet.C44 = C11, C12, C44

    return magnet

def get_2D_force(orientation):
    """Calculate central difference with traction multiplied by sign of surface
    normal.
    """
    return (np.array(pos_side_traction) + np.array(neg_side_traction)) / (cx, cy, cz)[orientation]

class TestBoundaryTraction2D:
    """A different stencil is used if the material is only 1 cell thick.
    Here we test that central difference for every orientation.
    """

    def test_x_side(self):
        orientation = 0
        magnet = get_2D_magnet(orientation)

        magnet.boundary_traction.neg_x_side = neg_side_traction
        magnet.boundary_traction.pos_x_side = pos_side_traction

        mxp_force = magnet.internal_body_force.eval()
        num_force = get_2D_force(orientation)

        for c in range(3):
            assert np.allclose(mxp_force[c], num_force[c], atol=1e-7*1e9*cx)

    def test_y_side(self):
        orientation = 1
        magnet = get_2D_magnet(orientation)

        magnet.boundary_traction.neg_y_side = neg_side_traction
        magnet.boundary_traction.pos_y_side = pos_side_traction

        mxp_force = magnet.internal_body_force.eval()
        num_force = get_2D_force(orientation)

        for c in range(3):
            assert np.allclose(mxp_force[c], num_force[c], atol=1e-7*1e9*cy)

    def test_z_side(self):
        orientation = 2
        magnet = get_2D_magnet(orientation)

        magnet.boundary_traction.neg_z_side = neg_side_traction
        magnet.boundary_traction.pos_z_side = pos_side_traction

        mxp_force = magnet.internal_body_force.eval()
        num_force = get_2D_force(orientation)

        for c in range(3):
            assert np.allclose(mxp_force[c], num_force[c], atol=1e-7*1e9*cz)
