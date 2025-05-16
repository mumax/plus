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
#include "magnetoelasticfield.hpp"
#include "zeeman.hpp"

Field evalEffectiveField(const Ferromagnet* magnet) {
  // there will probably be exchange, otherwise safely initialized as 0
  Field h = evalExchangeField(magnet);
  if (!anisotropyAssuredZero(magnet)) {h += evalAnisotropyField(magnet);}
  if (!externalFieldAssuredZero(magnet)) {h += evalExternalField(magnet);}
  if (!dmiAssuredZero(magnet)) {h += evalDmiField(magnet);}
  if (!demagFieldAssuredZero(magnet)) {h += evalDemagField(magnet);}
  if (!magnetoelasticAssuredZero(magnet)) {
      h += evalMagnetoelasticField(magnet);}
  if (magnet->isSublattice())
      if (!inHomoAfmExchangeAssuredZero(magnet)) {h += evalInHomogeneousAfmExchangeField(magnet);}
      if (!homoAfmExchangeAssuredZero(magnet)) {h += evalHomogeneousAfmExchangeField(magnet);}
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}
