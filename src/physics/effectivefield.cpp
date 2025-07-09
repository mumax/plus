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
#include "local_dmi.hpp"
#include "magnetoelasticfield.hpp"
#include "zeeman.hpp"

Field evalEffectiveField(const Ferromagnet* magnet) {
  // there will probably be exchange, otherwise safely initialized as 0
  Field h = evalExchangeField(magnet);
  if (!anisotropyAssuredZero(magnet)) {h += evalAnisotropyField(magnet);}
  if (!externalFieldAssuredZero(magnet)) {h += evalExternalField(magnet);}
  if (!inhomoDmiAssuredZero(magnet)) {h += evalDmiField(magnet);}
  if (!demagFieldAssuredZero(magnet)) {h += evalDemagField(magnet);}
  if (!magnetoelasticAssuredZero(magnet)) {
      h += evalMagnetoelasticField(magnet);}
  if (magnet->isSublattice())
      // AFM exchange terms
      if (!inHomoAfmExchangeAssuredZero(magnet)) {h += evalInHomogeneousAfmExchangeField(magnet);}
      if (!homoAfmExchangeAssuredZero(magnet)) {h += evalHomogeneousAfmExchangeField(magnet);}
      // Homogeneous (local) DMI term
      if (!homoDmiAssuredZero(magnet)) {h += evalHomoDmiField(magnet);}
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}
