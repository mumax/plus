#include <memory>
#include <stdexcept>

#include "afmexchange.hpp"
#include "altermagnet.hpp"
#include "dmi.hpp"
#include "energy.hpp"
#include "fieldquantity.hpp"
#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "neel.hpp"
#include "parameter.hpp"
#include "fullmag.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_altermagnet(py::module& m) {
  py::class_<Altermagnet, Magnet>(m, "Altermagnet")
      .def("sub1", &Altermagnet::sub1, py::return_value_policy::reference)
      .def("sub2", &Altermagnet::sub2, py::return_value_policy::reference)
      .def("sublattices", &Altermagnet::sublattices, py::return_value_policy::reference)
      .def("other_sublattice",
          [](const Altermagnet* m, Ferromagnet* mag) { return m->getOtherSublattices(mag)[0]; },
            py::return_value_policy::reference)
      .def_readonly("atmex_cell", &Altermagnet::afmex_cell)
      .def_readonly("atmex_nn", &Altermagnet::afmex_nn)
      .def_readonly("alterex_1", &Altermagnet::alterex_1)
      .def_readonly("alterex_2", &Altermagnet::alterex_2)
      .def_readonly("inter_alterex_1", &Altermagnet::interAlterex_1)
      .def_readonly("scale_alterex_1", &Altermagnet::scaleAlterex_1)
      .def_readonly("inter_alterex_2", &Altermagnet::interAlterex_2)
      .def_readonly("scale_alterex_2", &Altermagnet::scaleAlterex_2)
      .def_readonly("alterex_angle", &Altermagnet::alterex_angle)
      .def_readonly("inter_atmex_nn", &Altermagnet::interAfmExchNN)
      .def_readonly("scale_atmex_nn", &Altermagnet::scaleAfmExchNN)
      .def_readonly("latcon", &Altermagnet::latcon)
      .def_readonly("dmi_tensor", &Altermagnet::dmiTensor)
      .def_readonly("dmi_vector", &Altermagnet::dmiVector)

      .def("minimize", &Altermagnet::minimize, py::arg("tol"), py::arg("nsamples"))
      .def("relax", &Altermagnet::relax, py::arg("tol"));
      
  m.def("neel_vector",
        py::overload_cast<const Altermagnet*> (&neelVectorQuantity));
  m.def("full_magnetization",
        py::overload_cast<const Altermagnet*>(&fullMagnetizationQuantity));

  m.def("angle_field",
        py::overload_cast<const Altermagnet*>(&angleFieldQuantity));
  m.def("max_intracell_angle",
        py::overload_cast<const Altermagnet*>(&maxAngle));

  m.def("total_energy_density",
      [](const Altermagnet* m) {return totalEnergyDensityQuantity(m);});
  m.def("total_energy",
      [](const Altermagnet* m) {return totalEnergyQuantity(m);});
}