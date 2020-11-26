#include <memory>

#include "system.hpp"
#include "wrappers.hpp"

void wrap_system(py::module& m) {
  py::class_<System, std::shared_ptr<System>>(m, "System")
      .def_property_readonly("grid", &System::grid)
      .def_property_readonly("cellsize", &System::cellsize)
      .def("cell_position", &System::cellPosition);
}