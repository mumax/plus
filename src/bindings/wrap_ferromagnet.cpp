#include <memory>
#include <stdexcept>

#include "anisotropy.hpp"
#include "conductivitytensor.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "effectivefield.hpp"
#include "electricalpotential.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "fieldquantity.hpp"
#include "mumaxworld.hpp"
#include "neel.hpp"
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
      .def_property_readonly("system", &Ferromagnet::system)
      .def_property_readonly("comp", &Ferromagnet::ncomp)

      // TODO: implement the world property which returns the MumaxWorld to
      // which the ferromagnet belongs
      // .def_property_readonly("world",...)

      .def_property_readonly("magnetization", &Ferromagnet::magnetization)

      .def_readwrite("enable_demag", &Ferromagnet::enableDemag)
      .def_readwrite("enable_openbc", &Ferromagnet::enableOpenBC)
      .def_readwrite("bias_magnetic_field", &Ferromagnet::biasMagneticField,
                     "uniform external magnetic field")

      .def_readonly("msat", &Ferromagnet::msat)
      .def_readonly("msat2", &Ferromagnet::msat2)
      .def_readonly("alpha", &Ferromagnet::alpha)
      .def_readonly("ku1", &Ferromagnet::ku1)
      .def_readonly("ku12", &Ferromagnet::ku12)
      .def_readonly("ku2", &Ferromagnet::ku2)
      .def_readonly("ku22", &Ferromagnet::ku22)
      .def_readonly("kc1", &Ferromagnet::kc1)
      .def_readonly("kc2", &Ferromagnet::kc2)
      .def_readonly("kc3", &Ferromagnet::kc3)
      .def_readonly("kc12", &Ferromagnet::kc12)
      .def_readonly("kc22", &Ferromagnet::kc22)
      .def_readonly("kc32", &Ferromagnet::kc32)
      .def_readonly("aex", &Ferromagnet::aex)
      .def_readonly("aex2", &Ferromagnet::aex2)
      .def_readonly("afmex_cell", &Ferromagnet::afmex_cell)
      .def_readonly("afmex_nn", &Ferromagnet::afmex_nn)
      .def_readonly("anisU", &Ferromagnet::anisU)
      .def_readonly("anisC1", &Ferromagnet::anisC1)
      .def_readonly("anisC2", &Ferromagnet::anisC2)
      .def_readonly("latcon", &Ferromagnet::latcon)
      .def_readonly("Lambda", &Ferromagnet::Lambda)
      .def_readonly("FreeLayerThickness", &Ferromagnet::FreeLayerThickness)
      .def_readonly("eps_prime", &Ferromagnet::eps_prime)
      .def_readonly("FixedLayer", &Ferromagnet::FixedLayer)
      .def_readonly("xi", &Ferromagnet::xi)
      .def_readonly("pol", &Ferromagnet::pol)
      .def_readonly("jcur", &Ferromagnet::jcur)
      .def_readonly("temperature", &Ferromagnet::temperature)
      .def_readonly("dmi_tensor", &Ferromagnet::dmiTensor)
      .def_readonly("applied_potential", &Ferromagnet::appliedPotential)
      .def_readonly("conductivity", &Ferromagnet::conductivity)
      .def_readonly("amr_ratio", &Ferromagnet::amrRatio)
      .def_readonly("amr_ratio2", &Ferromagnet::amrRatio2)
      .def_readonly("RelaxTorqueThreshold", &Ferromagnet::RelaxTorqueThreshold)
      .def_readonly("poisson_system", &Ferromagnet::poissonSystem)
      
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
           py::arg("nsamples") = 10)
      .def("relax", &Ferromagnet::relax);

  m.def("torque", &torqueQuantity);
  m.def("llg_torque", &llgTorqueQuantity);
  m.def("spin_transfer_torque", &spinTransferTorqueQuantity);

  m.def("neel_vector", &neelVectorQuantity);

  m.def("demag_field", &demagFieldQuantity);
  m.def("demag_energy_density", &demagEnergyDensityQuantity);
  m.def("demag_energy", [](const Ferromagnet* magnet, const bool sub2) {return demagEnergyQuantity(magnet, sub2);});

  m.def("anisotropy_field", &anisotropyFieldQuantity);
  m.def("anisotropy_energy_density", &anisotropyEnergyDensityQuantity);
  //m.def("anisotropy_energy", &anisotropyEnergyQuantity);
  m.def("anisotropy_energy", [](const Ferromagnet* magnet, const bool sub2) {return anisotropyEnergyQuantity(magnet, sub2);});
  m.def("anisotropy_energy2", [](const Ferromagnet* magnet, const bool sub2) {return anisotropyEnergyQuantity(magnet, sub2);});


  m.def("exchange_field", &exchangeFieldQuantity);
  m.def("exchange_energy_density", &exchangeEnergyDensityQuantity);
  //m.def("exchange_energy", &exchangeEnergyQuantity);
  m.def("exchange_energy", [](const Ferromagnet* magnet, const bool sub2) {return exchangeEnergyQuantity(magnet, sub2);});
  m.def("exchange_energy2", [](const Ferromagnet* magnet, const bool sub2) {return exchangeEnergyQuantity(magnet, sub2);});
  m.def("max_angle", [](const Ferromagnet* magnet, const bool sub2) {return maxAngle(magnet, sub2);});

  m.def("dmi_field", &dmiFieldQuantity);
  m.def("dmi_energy_density", &dmiEnergyDensityQuantity);
  //m.def("dmi_energy", &dmiEnergyQuantity);
  m.def("dmi_energy", [](const Ferromagnet* magnet, const bool sub2) {return dmiEnergyQuantity(magnet, sub2);});
  m.def("dmi_energy2", [](const Ferromagnet* magnet, const bool sub2) {return dmiEnergyQuantity(magnet, sub2);});

  m.def("external_field", &externalFieldQuantity);
  m.def("zeeman_energy_density", &zeemanEnergyDensityQuantity);
//  m.def("zeeman_energy", &zeemanEnergyQuantity);
  m.def("zeeman_energy", [](const Ferromagnet* magnet, const bool sub2) {return zeemanEnergyQuantity(magnet, sub2);});
  m.def("zeeman_energy2", [](const Ferromagnet* magnet, const bool sub2) {return zeemanEnergyQuantity(magnet, sub2);});

  m.def("effective_field", &effectiveFieldQuantity);
  m.def("total_energy_density", &totalEnergyDensityQuantity);
  //m.def("total_energy", &totalEnergyQuantity);
  m.def("total_energy", [](const Ferromagnet* magnet, const bool sub2) {return totalEnergyQuantity(magnet, sub2);});
  m.def("total_energy2", [](const Ferromagnet* magnet, const bool sub2) {return totalEnergyQuantity(magnet, sub2);});

  m.def("conductivity_tensor", &conductivityTensorQuantity);
  m.def("electrical_potential", &electricalPotentialQuantity);

  m.def("thermal_noise", &thermalNoiseQuantity);

  m.def("_demag_kernel", [](const Ferromagnet* fm) {
    Grid grid = fm->grid();
    real3 cellsize = fm->world()->cellsize();
    StrayFieldKernel demagKernel(grid, grid, fm->world());
    return fieldToArray(demagKernel.field());
  });
}