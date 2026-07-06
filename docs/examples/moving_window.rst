:nosearch:

Moving simulation window
===============

In this example we move a domain wall in a ferromagnet using a Zhang-Li STT. We let the simulation
window move together with the wall, keeping the domain wall centered in the simulation space.
Using this, we can virtually simulate an infitly long magnetic nanowire using a limited number
of simulation cells.

Note:
The moving window functionality only works properly if
- The magnet parameters are uniform
- The magnet has no geometry
- The magnet has no regions

.. literalinclude:: ../../examples/moving_window.py
  :language: python
  :lines: 13-

.. video:: ../images/moving_window.mp4
   :align: center