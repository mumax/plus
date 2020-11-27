#include <memory>
#include <stdexcept>

#include "anisotropy.hpp"
#include "demag.hpp"
#include "effectivefield.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "fieldquantity.hpp"
#include "interfacialdmi.hpp"
#include "mumaxworld.hpp"
#include "parameter.hpp"
#include "strayfieldkernel.hpp"
#include "stt.hpp"
#include "thermalnoise.hpp"
#include "torque.hpp"
#include "world.hpp"
#include "wrappers.hpp"
#include "zeeman.hpp"

void wrap_ferromagnet(py::module& m) {
  py::class_<Ferromagnet>(m, "Ferromagnet")
      .def_property_readonly("name", &Ferromagnet::name)
      .def_property_readonly("grid", &Ferromagnet::grid)
      .def_property_readonly("cellsize", &Ferromagnet::cellsize)

      // TODO: implement the world property which returns the MumaxWorld to
      // which the ferromagnet belongs
      // .def_property_readonly("world",...)

      .def_property_readonly("magnetization", &Ferromagnet::magnetization)

      .def_readwrite("enable_demag", &Ferromagnet::enableDemag)

      .def_readonly("msat", &Ferromagnet::msat)
      .def_readonly("alpha", &Ferromagnet::alpha)
      .def_readonly("ku1", &Ferromagnet::ku1)
      .def_readonly("ku2", &Ferromagnet::ku2)
      .def_readonly("aex", &Ferromagnet::aex)
      .def_readonly("anisU", &Ferromagnet::anisU)
      .def_readonly("idmi", &Ferromagnet::idmi)
      .def_readonly("xi", &Ferromagnet::xi)
      .def_readonly("pol", &Ferromagnet::pol)
      .def_readonly("jcur", &Ferromagnet::jcur)
      .def_readonly("temperature", &Ferromagnet::temperature)

      .def(
          "magnetic_field_from_magnet",
          [](const Ferromagnet* fm, Ferromagnet* magnet) {
            const StrayField* strayField = fm->getStrayField(magnet);
            if (!strayField)
              throw std::runtime_error(
                  "Can not compute the magnetic field of the magnet");
            return strayField;
          },
          py::return_value_policy::reference)

      .def("minimize", &Ferromagnet::minimize, py::arg("tol") = 1e-6,
           py::arg("nsamples") = 10);

  m.def("torque", &torqueQuantity);
  m.def("llg_torque", &llgTorqueQuantity);
  m.def("spin_transfer_torque", &spinTransferTorqueQuantity);

  m.def("demag_field", &demagFieldQuantity);
  m.def("demag_energy_density", &demagEnergyDensityQuantity);
  m.def("demag_energy", &demagEnergyQuantity);

  m.def("anisotropy_field", &anisotropyFieldQuantity);
  m.def("anisotropy_energy_density", &anisotropyEnergyDensityQuantity);
  m.def("anisotropy_energy", &anisotropyEnergyQuantity);

  m.def("exchange_field", &exchangeFieldQuantity);
  m.def("exchange_energy_density", &exchangeEnergyDensityQuantity);
  m.def("exchange_energy", &exchangeEnergyQuantity);
  m.def("max_angle", &maxAngle);

  m.def("interfacialdmi_field", &interfacialDmiFieldQuantity);
  m.def("interfacialdmi_energy_density", &interfacialDmiEnergyDensityQuantity);
  m.def("interfacialdmi_energy", &interfacialDmiEnergyQuantity);

  m.def("external_field", &externalFieldQuantity);
  m.def("zeeman_energy_density", &zeemanEnergyDensityQuantity);
  m.def("zeeman_energy", &zeemanEnergyQuantity);

  m.def("effective_field", &effectiveFieldQuantity);
  m.def("total_energy_density", &totalEnergyDensityQuantity);
  m.def("total_energy", &totalEnergyQuantity);

  m.def("thermal_noise", &thermalNoiseQuantity);

  m.def("_demag_kernel", [](const Ferromagnet* fm) {
    Grid grid = fm->grid();
    real3 cellsize = fm->world()->cellsize();
    StrayFieldKernel demagKernel(grid, grid, cellsize);
    return fieldToArray(demagKernel.field());
  });
}
