:nosearch:

API
===

.. list-table:: Environment Variables
   :header-rows: 1

   * - Environment variable
     - Optional value
     - Meaning
   * - GPU_ABORT
     - "OFF"
     - Continue after a CUDA error instead of exiting (the default).
   * - MUMAXPLUS_FP_PRECISION
     - "SINGLE"/"1"/"32" or "DOUBLE"/"2"/"64"
     - Set the floating-point precision. See this :doc:`tutorial <tutorial/precision>`.


.. currentmodule:: mumaxplus

.. toctree::
   :maxdepth: 2

.. autosummary::
   :toctree: _api/mumaxplus

   World
   Grid
   Magnet
   Ferromagnet
   Antiferromagnet
   NcAfm
   Altermagnet
   TimeSolver
   ScalarQuantity
   FieldQuantity
   Variable
   Parameter
   InterParameter
   StrayField
   PoissonSystem
   DmiTensor
   BoundaryTraction
   Window

.. toctree::
   :maxdepth: 1

   util