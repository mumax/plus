#include <string>

#include "ferromagnet.hpp"
#include "grid.hpp"
#include "mumaxworld.hpp"
#include "wrappers.hpp"

void wrap_world(py::module& m) {
  // TODO: avoid destructor being called when python ref out of scope

  // in mumax5 module, MumaxWorld is the World
  py::class_<MumaxWorld>(m, "World")

      .def(py::init<real3, Grid>(), py::arg("cellsize"),
           py::arg("mastergrid") = Grid(int3{0, 0, 0}),
           "construct a World with a given cellsize, and optionally a "
           "mastergrid which defines a periodic simulation box")
      .def_property_readonly("cellsize", &MumaxWorld::cellsize,
                             "the cellsize of the world")
      .def_property_readonly("mastergrid", &MumaxWorld::mastergrid,
                             "mastergrid of the world")
      .def_readwrite("bias_magnetic_field", &MumaxWorld::biasMagneticField,
                     "uniform external magnetic field")
      .def("add_ferromagnet", &MumaxWorld::addFerromagnet, py::arg("grid"),
           py::arg("name") = std::string(""), "add a ferromagnet to the world",
           py::return_value_policy::reference)
      .def("get_ferromagnet", &MumaxWorld::getFerromagnet, py::arg("name"),
           "get a reference to a magnet by name",
           py::return_value_policy::reference);
}
