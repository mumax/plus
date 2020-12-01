#include <pybind11/numpy.h>

#include <string>

#include "ferromagnet.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "mumaxworld.hpp"
#include "system.hpp"
#include "timesolver.hpp"
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

      .def(
          "add_ferromagnet",
          [](MumaxWorld* world, Grid grid, std::string name) {
            return world->addFerromagnet(grid, name);
          },
          py::arg("grid"), py::arg("name") = std::string(""),
          py::return_value_policy::reference)

      .def(
          "add_ferromagnet",
          [](MumaxWorld* world, Grid grid, py::array_t<bool> geometryArray,
             std::string name) {
            py::buffer_info buf = geometryArray.request();
            GpuBuffer<bool> geometry(buf.size,
                                     reinterpret_cast<bool*>(buf.ptr));
            return world->addFerromagnet(grid, geometry, name);
          },
          py::arg("grid"), py::arg("geometry"),
          py::arg("name") = std::string(""), py::return_value_policy::reference)

      .def("get_ferromagnet", &MumaxWorld::getFerromagnet, py::arg("name"),
           "get a reference to a magnet by name",
           py::return_value_policy::reference)

      .def_property_readonly("timesolver", &MumaxWorld::timesolver,
                             py::return_value_policy::reference);
}
