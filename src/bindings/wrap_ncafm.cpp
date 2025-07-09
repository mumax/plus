#include <memory>
#include <stdexcept>

#include "ncafm.hpp"
#include "energy.hpp"
#include "fieldquantity.hpp"
#include "fullmag.hpp"
#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "ncafmangle.hpp"
#include "octupole.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_ncafm(py::module& m) {
  py::class_<NcAfm, Magnet>(m, "NcAfm")
      .def("sub1", &NcAfm::sub1, py::return_value_policy::reference)
      .def("sub2", &NcAfm::sub2, py::return_value_policy::reference)
      .def("sub3", &NcAfm::sub3, py::return_value_policy::reference)
      .def("sublattices", &NcAfm::sublattices, py::return_value_policy::reference)
      .def("other_sublattices",
          [](const NcAfm* m, Ferromagnet* mag) { return m->getOtherSublattices(mag); },
            py::return_value_policy::reference)

      .def_readonly("ncafmex_cell", &NcAfm::afmex_cell)
      .def_readonly("ncafmex_nn", &NcAfm::afmex_nn)
      .def_readonly("inter_ncafmex_nn", &NcAfm::interAfmExchNN)
      .def_readonly("scale_ncafmex_nn", &NcAfm::scaleAfmExchNN)
      .def_readonly("latcon", &NcAfm::latcon)
      .def_readonly("dmi_tensor", &NcAfm::dmiTensor)
      .def_readonly("dmi_vector", &NcAfm::dmiVector)

      .def("minimize", &NcAfm::minimize, py::arg("tol"), py::arg("nsamples"))
      .def("relax", &NcAfm::relax, py::arg("tol"));

  m.def("octupole_vector", &octupoleVectorQuantity);
  m.def("full_magnetization",
        py::overload_cast<const NcAfm*>(&fullMagnetizationQuantity));

  m.def("angle_field", &angleFieldQuantity);
  m.def("max_intracell_angle_between",
          [](const Ferromagnet* i, const Ferromagnet* j) { return evalMaxAngle(i, j); },
          py::arg("sub1"), py::arg("sub2"));

  m.def("total_energy_density",
        py::overload_cast<const NcAfm*>(&totalEnergyDensityQuantity));
  m.def("total_energy",
        py::overload_cast<const NcAfm*>(&totalEnergyQuantity));
}