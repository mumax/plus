#include "effectivefield.hpp"

#include "anisotropy.hpp"
#include "bulkdmi.hpp"
#include "demag.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "interfacialdmi.hpp"
#include "zeeman.hpp"

Field evalEffectiveField(const Ferromagnet* magnet) {
  Field h = evalDemagField(magnet);
  h += evalAnisotropyField(magnet);
  h += evalExchangeField(magnet);
  h += evalExternalField(magnet);
  h += evalInterfacialDmiField(magnet);
  h += evalBulkDmiField(magnet);
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field",
                          "T");
}
