:nosearch:

gui
===

A web GUI for mumax⁺. It runs a small web server
from your script. In the browser page you can watch the magnetization while it
evolves, plot table columns, change parameters, the solver and the initial
magnetization, paint on the magnet, add excitations, save checkpoints, record
movies and type Python in a console. The server only needs the Python standard
library. See the :doc:`GUI tutorial <tutorial/gui>` to get started.

.. currentmodule:: mumaxplus.gui

.. autosummary::
   :toctree: _api/gui
   :caption: gui

   GUI
   serve

Console commands
----------------

Next to the variables in ``namespace``, the GUI console provides these
simulation commands. They also work in the script once the :class:`GUI` has
been created, as long as ``namespace=globals()`` was passed.

.. list-table::
   :header-rows: 1

   * - Command
     - Meaning
   * - ``Run(t=None)``
     - Run for ``t`` seconds of simulation time, or until ``Break()`` when
       ``t`` is omitted.
   * - ``Steps(n)``
     - Take ``n`` time steps.
   * - ``Break()``
     - Stop the current ``Run`` or ``Steps``.
   * - ``Relax()``
     - Relax the magnetization to an energy minimum (``world.relax()``).
   * - ``Minimize()``
     - Minimize the energy (``world.minimize()``).
   * - ``TableAdd(quantity, name=None, unit=None)``
     - Same as :meth:`GUI.table_add`.
   * - ``TableAutosave(period)``
     - Same as :meth:`GUI.table_autosave`.
   * - ``TableSave()``
     - Write one table row to ``table.txt`` now.
   * - ``SetSolver(method)``
     - Change the Runge-Kutta method of the time solver.
