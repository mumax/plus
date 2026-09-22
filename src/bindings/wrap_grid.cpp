#include "grid.hpp"
#include "wrappers.hpp"

void wrap_grid(py::module& m) {
  py::class_<Grid>(m, "Grid", "TODO: add Grid class documentation")

      .def(py::init<int3, int3>(), "Construct a grid with a given size and origin",
           py::arg("size"), py::arg("origin") = int3{0, 0, 0})
      .def_property_readonly("size", &Grid::size, "size of the grid")
      .def_property_readonly("origin", &Grid::origin, "origin of the grid")
      .def_property_readonly("ncells", &Grid::ncells,
                             "total number of cells in the grid")
      .def(py::self == py::self)
      .def(py::self != py::self);
}
