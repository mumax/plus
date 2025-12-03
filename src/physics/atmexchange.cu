#include "altermagnet.hpp"
#include "cudalaunch.hpp"
#include "dmi.hpp" // used for Neumann BC
#include "energy.hpp"
#include "atmexchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "inter_parameter.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool atmExchangeAssuredZero(const Ferromagnet* magnet) {
  if (!magnet->hostMagnet()->asATM()) { return true; }
  if (magnet->hostMagnet()->asATM()->atmex_ani.assuredZero() ||
      magnet->msat.assuredZero()) { return true; }

  for (auto sub : magnet->hostMagnet()->getOtherSublattices(magnet)) {
    if (!sub->msat.assuredZero())
      return false;
  }
  return true;
}

// ATM exchange between NN cells
__global__ void k_atmExchangeField(CuField hField,
                                const CuField mField,
                                const CuParameter atmex_ani,
                                const CuParameter msat,
                                const Grid mastergrid,
                                const real fac) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const Grid grid = mField.system.grid;
  if (!grid.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real3 m = mField.vectorAt(idx);
  const real atmex = atmex_ani.valueAt(idx);

  int3 xm = int3{-1, 0, 0};
  int3 xp = int3{+1, 0, 0};
  int3 ym = int3{0, -1, 0};
  int3 yp = int3{0, +1, 0};

  //int3 xm2 = int3{-2, 0, 0};
  //int3 xp2 = int3{+2, 0, 0};
  //int3 ym2 = int3{0, -2, 0};
  //int3 yp2 = int3{0, +2, 0};

  real hx = system.cellsize.x;
  real hy = system.cellsize.y;

  // four diagonal neighbors
  const int3 c_xp_yp = mastergrid.wrap(coo + xp + yp);
  const int3 c_xp_ym = mastergrid.wrap(coo + xp + ym);
  const int3 c_xm_yp = mastergrid.wrap(coo + xm + yp);
  const int3 c_xm_ym = mastergrid.wrap(coo + xm + ym);

  //int3 c_xp2_yp2 = mastergrid.wrap(coo + xp2 + yp2);
  //int3 c_xp2_ym2 = mastergrid.wrap(coo + xp2 + ym2);
  //int3 c_xm2_yp2 = mastergrid.wrap(coo + xm2 + yp2);
  //int3 c_xm2_ym2 = mastergrid.wrap(coo + xm2 + ym2);

  real3 m_xp_yp = hField.cellInGeometry(c_xp_yp) ? mField.vectorAt(c_xp_yp) : real3{0, 0, 0};
  real3 m_xp_ym = hField.cellInGeometry(c_xp_ym) ? mField.vectorAt(c_xp_ym) : real3{0, 0, 0};
  real3 m_xm_yp = hField.cellInGeometry(c_xm_yp) ? mField.vectorAt(c_xm_yp) : real3{0, 0, 0};
  real3 m_xm_ym = hField.cellInGeometry(c_xm_ym) ? mField.vectorAt(c_xm_ym) : real3{0, 0, 0};

  //real3 m_xp2_yp2 = hField.cellInGeometry(c_xp2_yp2) ? mField.vectorAt(c_xp2_yp2) : real3{0, 0, 0};
  //real3 m_xp2_ym2 = hField.cellInGeometry(c_xp2_ym2) ? mField.vectorAt(c_xp2_ym2) : real3{0, 0, 0};
  //real3 m_xm2_yp2 = hField.cellInGeometry(c_xm2_yp2) ? mField.vectorAt(c_xm2_yp2) : real3{0, 0, 0};
  //real3 m_xm2_ym2 = hField.cellInGeometry(c_xm2_ym2) ? mField.vectorAt(c_xm2_ym2) : real3{0, 0, 0};

  real check = 1.0;
  if (m_xp_yp == real3{0, 0, 0}) { check = 0.0; }
  if (m_xp_ym == real3{0, 0, 0}) { check = 0.0; }
  if (m_xm_yp == real3{0, 0, 0}) { check = 0.0; }
  if (m_xm_ym == real3{0, 0, 0}) { check = 0.0; }
  //if (m_xp2_yp2 == real3{0, 0, 0}) { check = 0.0; }
  //if (m_xp2_ym2 == real3{0, 0, 0}) { check = 0.0; }
  //if (m_xm2_yp2 == real3{0, 0, 0}) { check = 0.0; }
  //if (m_xm2_ym2 == real3{0, 0, 0}) { check = 0.0; }

  real denom = hx * hy;
  real3 dxy_m_first = (1.0/4.0) * ((m_xp_yp + m_xm_ym) - (m_xp_ym + m_xm_yp) );
  //real3 dxy_m_second = (1.0/48.0) * (m_xp2_yp2 + m_xm2_ym2 - m_xm2_yp2 - m_xp2_ym2);

  real3 deriv = (dxy_m_first) * check / denom;
  real3 result = 2.0 * fac * atmex * deriv;
  hField.setVectorInCell(idx, result / msat.valueAt(idx));
}

Field evalAtmExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3, real3{0, 0, 0});

  if (atmExchangeAssuredZero(magnet))
    return hField;

  auto mag = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();

  auto host = magnet->hostMagnet();
  auto atmex_ani = host->asATM()->atmex_ani.cu();

  int i = host->getSublatticeIndex(magnet);
  real fac;
  if (i == 0) { fac = 1.0; }
  else { fac = -1.0; }

  cudaLaunch(hField.grid().ncells(), k_atmExchangeField, hField.cu(),
              mag, atmex_ani, msat, magnet->world()->mastergrid(), fac);
  return hField;
}

Field evalAtmExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (atmExchangeAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalAtmExchangeField(magnet), 0.5);
}

real evalAtmExchangeEnergy(const Ferromagnet* magnet) {
  if (atmExchangeAssuredZero(magnet))
    return 0;

  real edens = atmExchangeEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

FM_FieldQuantity atmExchangeFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAtmExchangeField, 3,
                          "anisotropic_exchange_field", "T");
}

FM_FieldQuantity atmExchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAtmExchangeEnergyDensity, 1,
                          "anisotropic_exchange_energy_density", "J/m3");
}

FM_ScalarQuantity atmExchangeEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalAtmExchangeEnergy,
                          "anisotropic_exchange_energy", "J");
}