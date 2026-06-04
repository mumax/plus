#include "traction.hpp"
#include "wrappers.hpp"

void wrap_traction(py::module& m) {
  py::class_<BoundaryTraction>(m, "BoundaryTraction")
      .def_readonly("pos_x_side", &BoundaryTraction::posXside)
      .def_readonly("neg_x_side", &BoundaryTraction::negXside)
      .def_readonly("pos_y_side", &BoundaryTraction::posYside)
      .def_readonly("neg_y_side", &BoundaryTraction::negYside)
      .def_readonly("pos_z_side", &BoundaryTraction::posZside)
      .def_readonly("neg_z_side", &BoundaryTraction::negZside);
}
