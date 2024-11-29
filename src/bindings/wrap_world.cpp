#include <pybind11/numpy.h>

#include <string>

#include "antiferromagnet.hpp"
#include "ferromagnet.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "mumaxworld.hpp"
#include "system.hpp"
#include "timesolver.hpp"
#include "wrappers.hpp"

/* Helper function to add any magnet instance to the world*/
template<typename FuncType>
auto add_magnet(MumaxWorld* world,
                Grid grid,
                py::object geometryArray,
                py::object regionsArray,
                std::string& name,
                FuncType addFunc) {
     GpuBuffer<bool> geometry;
     GpuBuffer<uint> regions;

     if (!py::isinstance<py::none>(geometryArray)) {
          py::buffer_info buf = geometryArray.cast<py::array_t<bool>>().request();
          geometry = GpuBuffer<bool>(buf.size, reinterpret_cast<bool*>(buf.ptr));
     }
     if (!py::isinstance<py::none>(regionsArray)) {
          py::buffer_info buf = regionsArray.cast<py::array_t<uint>>().request();
          regions = GpuBuffer<uint>(buf.size, reinterpret_cast<uint*>(buf.ptr));
     }
     return (world->*addFunc)(grid, geometry, regions, name);
}

void wrap_world(py::module& m) {
  // in mumax⁺ module, MumaxWorld is the World
  py::class_<MumaxWorld>(m, "World")

      .def(py::init<real3>(), py::arg("cellsize"),
           "construct a World with a given cellsize")
      .def(py::init<real3, Grid, int3>(), py::arg("cellsize"),
           py::arg("mastergrid"), py::arg("pbc_repetitions"),
           "construct a World with a given cellsize, mastergrid and "
           "pbcRepetitions which define a periodic simulation box")
      .def_property_readonly("cellsize", &MumaxWorld::cellsize,
                             "the cellsize of the world")
      .def_readwrite("bias_magnetic_field", &MumaxWorld::biasMagneticField,
                     "uniform external magnetic field")
      .def_readwrite("RelaxTorqueThreshold", &MumaxWorld::RelaxTorqueThreshold)

      .def("add_ferromagnet",
          [](MumaxWorld* world, Grid grid, py::object geometryArray=py::none(),
             py::object regionsArray=py::none(), std::string name="") {
               return add_magnet(world, grid,
                                 geometryArray,
                                 regionsArray,
                                 name,
                                 &MumaxWorld::addFerromagnet);

          },
          py::arg("grid"), py::arg("geometry_array")=py::none(),
          py::arg("regions_array")=py::none(), py::arg("name") = std::string(""),
          py::return_value_policy::reference_internal)

      .def("add_antiferromagnet",
          [](MumaxWorld* world, Grid grid, py::object geometryArray=py::none(),
             py::object regionsArray=py::none(), std::string name="") {
               return add_magnet(world, grid,
                                 geometryArray,
                                 regionsArray,
                                 name,
                                 &MumaxWorld::addAntiferromagnet);
          },
          py::arg("grid"), py::arg("geometry_array")=py::none(),
          py::arg("regions_array")=py::none(), py::arg("name") = std::string(""),
          py::return_value_policy::reference_internal)

      .def("get_ferromagnet", &MumaxWorld::getFerromagnet, py::arg("name"),
           "get a reference to a ferromagnet by name",
           py::return_value_policy::reference)

      .def("get_antiferromagnet", &MumaxWorld::getAntiferromagnet, py::arg("name"),
           "get a reference to an antiferromagnet by name",
           py::return_value_policy::reference)

      .def_property_readonly("ferromagnets", &MumaxWorld::ferromagnets,
           "get a map of all ferromagnets in this world")

      .def_property_readonly("antiferromagnets", &MumaxWorld::antiferromagnets,
           "get a map of all antiferromagnets in this world")

      .def_property_readonly("timesolver", &MumaxWorld::timesolver,
                             py::return_value_policy::reference)
                             
      .def("minimize", &MumaxWorld::minimize, py::arg("tol"), py::arg("nsamples"))
      .def("relax", &MumaxWorld::relax, py::arg("tol"))

      // PBC
      .def_property_readonly("bounding_grid", &MumaxWorld::boundingGrid,
           "Returns grid which is the minimum bounding box of all magnets "
           "currently in the world.")
      .def("set_pbc", py::overload_cast<const Grid, const int3>
           (&MumaxWorld::setPBC), py::arg("mastergrid"),
           py::arg("pbc_repetitions"), "Set the PBC")
      .def("set_pbc", py::overload_cast<const int3>(&MumaxWorld::setPBC),
           py::arg("pbc_repetitions"), "Set the PBC")
      .def("unset_pbc", &MumaxWorld::unsetPBC, "Unset the PBC")
      .def_property("mastergrid", &MumaxWorld::mastergrid,
                    &MumaxWorld::setMastergrid, "mastergrid of the world")
      .def_property("pbc_repetitions", &MumaxWorld::pbcRepetitions,
                    &MumaxWorld::setPbcRepetitions, "PBC repetitions of the world")
     ;
}
