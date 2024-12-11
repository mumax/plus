#include <memory>
#include <stdexcept>

#include "elasticdamping.hpp"
#include "elasticenergies.hpp"
#include "elastodynamics.hpp"
#include "elasticforce.hpp"
#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "poyntingvector.hpp"
#include "straintensor.hpp"
#include "strayfieldkernel.hpp"
#include "stresstensor.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_magnet(py::module& m) {
  py::class_<Magnet>(m, "Magnet")
      .def_property_readonly("name", &Magnet::name)
      .def_property_readonly("system", &Magnet::system)
      .def_property_readonly("world", &Magnet::mumaxWorld)

      .def_property_readonly("elastic_displacement",
                             &Magnet::elasticDisplacement)
      .def_property_readonly("elastic_velocity", &Magnet::elasticVelocity)

      .def_readwrite("enable_as_stray_field_source",
                     &Magnet::enableAsStrayFieldSource)
      .def_readwrite("enable_as_stray_field_destination",
                     &Magnet::enableAsStrayFieldDestination)
      .def_property("enable_elastodynamics",
                    &Magnet::enableElastodynamics,
                    &Magnet::setEnableElastodynamics)

      // elasticity parameters
      .def_readonly("external_body_force", &Magnet::externalBodyForce)
      .def_readonly("C11", &Magnet::C11)
      .def_readonly("C12", &Magnet::C12)
      .def_readonly("C44", &Magnet::C44)
      .def_readonly("eta", &Magnet::eta)
      .def_readonly("rho", &Magnet::rho)

      .def("stray_field_from_magnet",
          [](const Magnet* m, Magnet* magnet) {
            const StrayField* strayField = m->getStrayField(magnet);
            if (!strayField)
              throw std::runtime_error(
                  "Can not compute the magnetic field of the magnet");
            return strayField;
          },
          py::return_value_policy::reference)
  ;

  m.def("_demag_kernel", [](const Magnet* m) {
    StrayFieldKernel demagKernel(m->grid(), m->grid(), m->world());
    return fieldToArray(demagKernel.field());
  });

  // Elasticity
  m.def("strain_tensor", &strainTensorQuantity);
  m.def("stress_tensor", &stressTensorQuantity);

  m.def("elastic_force", &elasticForceQuantity);
  m.def("effective_body_force", &effectiveBodyForceQuantity);
  m.def("elastic_damping", &elasticDampingQuantity);
  m.def("elastic_acceleration", &elasticAccelerationQuantity);
  
  m.def("kinetic_energy_density", &kineticEnergyDensityQuantity);
  m.def("kinetic_energy", &kineticEnergyQuantity);
  m.def("elastic_energy_density", &elasticEnergyDensityQuantity);
  m.def("elastic_energy", &elasticEnergyQuantity);

  m.def("poynting_vector", &poyntingVectorQuantity);

}
