#include "wrappers.hpp"

PYBIND11_MODULE(_mumaxpluscpp, m) {
  wrap_fieldquantity(m);
  wrap_antiferromagnetfieldquantity(m);
  wrap_ferromagnetfieldquantity(m);
  wrap_magnetfieldquantity(m);
  wrap_scalarquantity(m);
  wrap_antiferromagnetscalarquantity(m);
  wrap_ferromagnetscalarquantity(m);
  wrap_magnetscalarquantity(m);
  wrap_magnet(m);
  wrap_antiferromagnet(m);
  wrap_ferromagnet(m);
  wrap_field(m);
  wrap_grid(m);
  wrap_parameter(m);
  wrap_timesolver(m);
  wrap_variable(m);
  wrap_world(m);
  wrap_poissonsolver(m);
  wrap_linsolver(m);
  wrap_strayfield(m);
  wrap_system(m);
  wrap_dmitensor(m);
  wrap_voronoi(m);
}
