#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "quantityevaluator.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "zeeman.hpp"

bool strayFieldsAssuredZero(const Ferromagnet* ferromagnet) {
  const Magnet* magnet;
  if (ferromagnet->isSublattice()) {
    magnet = ferromagnet->hostMagnet();
  } else {
    magnet = ferromagnet;
  }

  if (!magnet->enableAsStrayFieldDestination)
    return true;

  auto strayFields = magnet->getStrayFields();
  for (auto strayField : strayFields) {
    // Avoid the demag field, we only want external fields
    if (strayField->source() == magnet)
      continue;

    if (strayField->source()->enableAsStrayFieldSource &&
        !strayField->assuredZero()) {
      return false;
    }
  }
  return true;
}

bool worldBiasFieldAssuredZero(const Magnet* magnet) {
  real3 b_ext = magnet->mumaxWorld()->biasMagneticField;
  return b_ext == real3{0.0, 0.0, 0.0};
}

bool magnetBiasFieldAssuredZero(const Ferromagnet* magnet) {
  return magnet->biasMagneticField.assuredZero();
}

bool externalFieldAssuredZero(const Ferromagnet* magnet) {
  return (strayFieldsAssuredZero(magnet)
          && worldBiasFieldAssuredZero(magnet)
          && magnetBiasFieldAssuredZero(magnet));
}

Field evalExternalField(const Ferromagnet* magnet) {

  Field h(magnet->system(), 3);
  if (externalFieldAssuredZero(magnet)) {
    h.makeZero();
    return h;
  }

  real3 wB_bias = magnet->mumaxWorld()->biasMagneticField; // bias field on world
  auto& mB_bias = magnet->biasMagneticField; // bias field on individual magnet

  h.setUniformComponent(0, wB_bias.x);
  h.setUniformComponent(1, wB_bias.y);
  h.setUniformComponent(2, wB_bias.z);

  mB_bias.addToField(h);

  // stray fields
  const Magnet* dstMagnet;  // destination magnet of strayfields
  if (magnet->isSublattice()) {
    dstMagnet = magnet->hostMagnet();
  } else {
    dstMagnet = magnet;
  }
  if (dstMagnet->enableAsStrayFieldDestination) {  // only calculate if enabled
    std::vector<const StrayField*> strayFields = dstMagnet->getStrayFields();
    for (auto strayField : strayFields) {
      // Avoid the demag field, we only want external fields
      if (strayField->source() == dstMagnet)
        continue;
      // Avoid disabled stray fields
      if (!strayField->source()->enableAsStrayFieldSource)
        continue;
      strayField->addToField(h);
    }
  }

  return h;
}

Field evalZeemanEnergyDensity(const Ferromagnet* magnet) {
  if (externalFieldAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalExternalField(magnet), 1.0);
}

real zeemanEnergy(const Ferromagnet* magnet) {
  if (externalFieldAssuredZero(magnet))
    return 0.0;

  real edens = zeemanEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

FM_FieldQuantity externalFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalExternalField, 3, "external_field", "T");
}

FM_FieldQuantity zeemanEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalZeemanEnergyDensity, 1, "zeeman_energy_density", "J/m3");
}

FM_ScalarQuantity zeemanEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, zeemanEnergy, "zeeman_energy", "J");
}