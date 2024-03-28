#include "demag.hpp"

#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "world.hpp"

bool demagFieldAssuredZero(const Ferromagnet* magnet) {
  return !magnet->enableDemag || magnet->msat.assuredZero();
}

Field evalDemagField(const Ferromagnet* magnet) {
  if (demagFieldAssuredZero(magnet))
    return Field(magnet->system(), magnet->magnetization()->ncomp(), 0.0);
  return magnet->getStrayField(magnet)->eval();
}

Field evalDemagEnergyDensity(const Ferromagnet* magnet) {
  if (demagFieldAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalDemagField(magnet), 0.5);
}

real evalDemagEnergy(const Ferromagnet* magnet, const bool sub2) {
  if (demagFieldAssuredZero(magnet))
    return 0.0;
  int ncells = magnet->grid().ncells();
  real edensAverage = demagEnergyDensityQuantity(magnet).average()[0];
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}

FM_FieldQuantity demagFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDemagField, magnet->magnetization()->ncomp()
                          , "demag_field", "T");
}

FM_FieldQuantity demagEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDemagEnergyDensity, 1,
                          "demag_energy_density", "J/m3");
}

FM_ScalarQuantity demagEnergyQuantity(const Ferromagnet* magnet, const bool sub2) {
  std::string name = sub2 ? "demag_energy2" : "demag_energy";
  return FM_ScalarQuantity(magnet, evalDemagEnergy, sub2, name, "J");
}
