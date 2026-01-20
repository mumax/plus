#include <memory>
#include <stdexcept>

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
      .def_readonly("A1", &Altermagnet::A1)
      .def_readonly("A2", &Altermagnet::A2)
      .def_readonly("angle", &Altermagnet::angle)
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

  //m.def("angle_field", &angleFieldQuantity);
  //m.def("max_intracell_angle", &maxAngle);

  m.def("total_energy_density",
      [](const Altermagnet* m) {return totalEnergyDensityQuantity(m);});
  m.def("total_energy",
      [](const Altermagnet* m) {return totalEnergyQuantity(m);});
}