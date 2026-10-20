:nosearch:

Magnetoelastics
===============

In this example we initialize a magnet magnetized to the right.
The magnet is minimized and the elastic parameters are assigned together
with a circular area in which an external sinusoidal field is applied
in the y-direction. This simulation runs for 0.5 ns and returns an animation
of the y-magnetization and the amplified displacement.

.. literalinclude:: ../../examples/magnetoelastic.py
  :language: python
  :lines: 9-

.. video:: ../images/magnetoelastic.mp4
   :align: center