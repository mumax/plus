#include "wrappers.hpp"

PYBIND11_MODULE(_mumax5cpp, m) {
  wrap_fieldquantity(m);
  wrap_ferromagnetfieldquantity(m);
  wrap_scalarquantity(m);
  wrap_ferromagnetscalarquantity(m);
  wrap_debug(m);
  wrap_ferromagnet(m);
  wrap_field(m);
  wrap_grid(m);
  wrap_parameter(m);
  wrap_timesolver(m);
  wrap_variable(m);
  wrap_world(m);
  wrap_strayfield(m);
}