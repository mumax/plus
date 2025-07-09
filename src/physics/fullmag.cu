#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fullmag.hpp"
#include "ncafm.hpp"

Field evalFMFullMag(const Ferromagnet* magnet) {
  return magnet->msat.eval() * magnet->magnetization()->field();
}

Field evalHMFullMag(const HostMagnet* magnet) {
  // TODO: do we want one general evalfunc?
  auto sublattices = magnet->sublattices();
  Field result = sublattices[0]->msat.eval() * sublattices[0]->magnetization()->field();
  for (int i = 1; i < sublattices.size(); i++)
    addTo(result, sublattices[i]->msat.eval(), sublattices[i]->magnetization()->field());
  return result;
}

FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet* magnet) {
    return FM_FieldQuantity(magnet, evalFMFullMag, 3, "full_magnetization", "A/m");
}

AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalHMFullMag, 3, "full_magnetization", "A/m");
}

NcAfm_FieldQuantity fullMagnetizationQuantity(const NcAfm* magnet) {
    return NcAfm_FieldQuantity(magnet, evalHMFullMag, 3, "full_magnetization", "A/m");
}