"""This script compares the coercivity in the small particle limit of standard 
problem 2 to the analytical value. This mostly tests the minimize function.
TODO: try with relax function as well
Standard problems: https://www.ctcms.nist.gov/~rdm/mumag.org.html
Analytical value:
https://pubs.aip.org/aip/jap/article-abstract/87/9/5520/530184/Behavior-of-MAG-standard-problem-No-2-in-the-small
"""

import numpy as np

from mumaxplus import Ferromagnet, Grid, World

rHc_anal = 0.057069478  # analytical value of relative coercive field H_C/Msat


def test_stdp2_coercivity():

    # dimensions
    t_p_d = 0.1  # thickness/width ratio
    L_p_d = 5.0  # length/width ratio
    d = 0.1  # dimensionless width d = width/l_ex; close enough to infinitesimally small
    t = t_p_d * d
    L = L_p_d * d

    # taking realistic parameters, but does not matter
    msat = 800e3
    aex = 13e-12
    mu0 = 1.25663706212e-6
    l_ex = np.sqrt(2 * aex / (mu0 * msat**2))

    # single cell is enough
    world = World(cellsize=(L * l_ex, d * l_ex, t * l_ex))
    magnet = Ferromagnet(world, Grid((1, 1, 1)))
    magnet.msat = msat
    magnet.aex = aex

    # start with remnance magnetization
    magnet.magnetization = (1, 1, 1)  # fully saturated in specified direction
    rHmag_0 = 0  # but no H field
    magnet.minimize()
    # non-normalized magnetization along [1,1,1] direction
    m_0 = np.sum(magnet.magnetization.average())

    # now add H field in opposite direction
    H_direction = np.asarray((-1, -1, -1)) / np.sqrt(3)
    rH_min = 0.0500  # start of search fairly close to analytical value
    rHmag_1 = rH_min  # H field magnitude relative to Msat
    world.bias_magnetic_field = mu0 * rHmag_1 * msat * H_direction  # B field in Tesla
    magnet.minimize()
    m_1 = np.sum(magnet.magnetization.average())

    # keep raising the magnetic field magnitude until the magnet flips
    rH_step = 0.00005  # lower is better H_C resolution
    while m_1 > 0:
        # save previous attempts
        rHmag_0 = rHmag_1
        m_0 = m_1

        rHmag_1 += rH_step  # raise field magnitude
        world.bias_magnetic_field = mu0 * rHmag_1 * msat * H_direction
        magnet.minimize()  # TODO: try with relax

        m_1 = np.sum(magnet.magnetization.average())  # find m projection

    # linearly interpolate where mx+my+mz=0 => best relative coercive field H_C/Msat
    rHc_sim = (m_0 * rHmag_1 - m_1 * rHmag_0) / (m_0 - m_1)
    assert abs(rHc_anal - rHc_sim) < 1e-5
