#include "effectivefield.hpp"

#include "afmexchange.hpp"
#include "anisotropy.hpp"
#include "antiferromagnet.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "zeeman.hpp"

Field evalEffectiveField(const Ferromagnet* magnet) {
  Field h = evalAnisotropyField(magnet);
  h += evalExchangeField(magnet);
  h += evalExternalField(magnet);
  h += evalDmiField(magnet);
  h += evalDemagField(magnet);
  return h;
}

Field evalAFMEffectiveField(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  Field h = evalEffectiveField(sublattice);
  h += evalAFMExchangeField(magnet, sublattice);
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}

AFM_FieldQuantity AFM_effectiveFieldQuantity(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  return AFM_FieldQuantity(magnet, sublattice, evalAFMEffectiveField, 3, "effective_field", "T");
}