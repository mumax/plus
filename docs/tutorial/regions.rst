:nosearch:

Regions
=======

Material parameters of a magnet instance can be set by using a numpy array, 
allowing for inhomogeneities. This is very different compared to mumax³, where 
different regions had to be used. The benefit of defining regions comes into play
when there's a certain interaction between these different parts of a magnet, like the exchange interaction which couples nearest neighbouring simulation cells.

For this reason, mumax⁺ also provides an option to define regions inside a magnetic
material and set parameter values both in and between (where appropriate) different
regions.

.. code-block:: python
    
    import numpy as np

Parameters which can be set between different regions, also have an extra scaling
factor. If the parameter must not act between different regions, then this scaling
factor must be set to zero by the user. If the interregional parameter is not set,
then the harmonic mean of the parameter values of neighbouring cells is used.


Regions can be set in the same way that one would set the geometry of a magnet,
one of which is using a numpy array. This array contains integer values which
corresponds to region indices. This can be done as follows

.. code-block:: python
    
    from mumaxplus import Ferromagnet, Grid, World
    from mumaxplus.util import show_field

    world = World(cellsize=(1e-9, 1e-9, 1e-9))
    grid = Grid((5, 5, 1))
    regions = np.zeros(grid.shape)
    regions[:, :, :1] = 1
    regions[:, :, 1:2] = 2
    regions[:, :, 2:] = 3
    print(regions)

    magnet = Ferromagnet(world, grid, regions=regions)

.. code-block:: console
    
    [[[1. 2. 3. 3. 3.]
    [1. 2. 3. 3. 3.]
    [1. 2. 3. 3. 3.]
    [1. 2. 3. 3. 3.]
    [1. 2. 3. 3. 3.]]]

Here we have split up our magnet into 3 strips by defining three regions with
region indices 1, 2 and 3 (note that these indices can take on any integer value).
Now one can set parameter values in each region seperately.

.. code-block:: python
    
    # Set parameter values for all regions
    magnet.alpha = 0.1
    magnet.msat = 800e3
    magnet.enable_demag = False

    magnet.msat.set_in_region(2, 1e6) # Change msat in middle strip

    # Set exchange constant in regions seperately
    magnet.aex.set_in_region(1, 5e-12)
    magnet.aex.set_in_region(2, 13e-12)
    magnet.aex.set_in_region(3, 20e-12)

    # Set exchange constant between different regions
    magnet.scale_exchange.set_between(1, 2, 0) # No exchange between first two "strips" of the magnet
    magnet.inter_exchange.set_between(2, 3, 15e-12)

    show_field(magnet.magnetization)

    world.timesolver.run(.1e-9)

    show_field(magnet.magnetization)

.. image:: ../images/regions_1.png
   :width: 45%

.. image:: ../images/regions_2.png
   :width: 45%

Likewise, the antiferromagnetic nearest-neighbour exchange constant, ``afmex_nn``,
can be set in the same way.