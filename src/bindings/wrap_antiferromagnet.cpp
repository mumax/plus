#include <memory>
#include <stdexcept>

#include "afmexchange.hpp"
#include "antiferromagnet.hpp"
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

void wrap_antiferromagnet(py::module& m) {
  py::class_<Antiferromagnet, Magnet>(m, "Antiferromagnet")
      .def("sub1", &Antiferromagnet::sub1, py::return_value_policy::reference)
      .def("sub2", &Antiferromagnet::sub2, py::return_value_policy::reference)
      .def("sublattices", &Antiferromagnet::sublattices, py::return_value_policy::reference)
      .def("other_sublattice",
          [](const Antiferromagnet* m, Ferromagnet* mag) { return m->getOtherSublattices(mag)[0]; },
            py::return_value_policy::reference)
      .def_readonly("afmex_cell", &Antiferromagnet::afmex_cell)
      .def_readonly("afmex_nn", &Antiferromagnet::afmex_nn)
      .def_readonly("inter_afmex_nn", &Antiferromagnet::interAfmExchNN)
      .def_readonly("scale_afmex_nn", &Antiferromagnet::scaleAfmExchNN)
      .def_readonly("latcon", &Antiferromagnet::latcon)
      .def_readonly("dmi_tensor", &Antiferromagnet::dmiTensor)
      .def_readonly("dmi_vector", &Antiferromagnet::dmiVector)

      .def("minimize", &Antiferromagnet::minimize, py::arg("tol"), py::arg("nsamples"))
      .def("relax", &Antiferromagnet::relax, py::arg("tol"));
      
  m.def("neel_vector", &neelVectorQuantity);
  m.def("full_magnetization",
        py::overload_cast<const Antiferromagnet*>(&fullMagnetizationQuantity));

  m.def("angle_field", &angleFieldQuantity);
  m.def("max_intracell_angle", &maxAngle);

  m.def("total_energy_density",
        py::overload_cast<const Antiferromagnet*>(&totalEnergyDensityQuantity));
  m.def("total_energy",
        py::overload_cast<const Antiferromagnet*>(&totalEnergyQuantity));
}