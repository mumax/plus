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
  if ((magnet->hostMagnet()->asATM()->A1.assuredZero() &&
       magnet->hostMagnet()->asATM()->A2.assuredZero()) ||
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
                                const CuParameter A1,
                                const CuParameter A2,
                                const CuParameter angle,
                                const CuParameter msat,
                                const Grid mastergrid,
                                const real3 w) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
  const real a1 = A1.valueAt(idx);
  const real a2 = A2.valueAt(idx);

  // CALCULATE PREFACTORS
  real c = cos(angle.valueAt(idx));
  real s = sin(angle.valueAt(idx));
  real c2 = c * c;
  real s2 = s * s;
  real Cxx = a1 * c2 + a2 * s2;
  real Cyy = a2 * c2 + a1 * s2;
  real Cxy = 2 * c * s * (a1 - a2);

  real3 C = {Cxx, Cyy, Cxy};

  real3 h{0, 0, 0};

  // SECOND ORDER DERIVATIVES
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0}, int3{0, 1, 0}}) {
    const int3 coo_ = mastergrid.wrap(coo + rel_coo);
    const int idx_ = grid.coord2index(coo_);

    real3 m_;
    int3 normal = rel_coo * rel_coo;

    if(hField.cellInGeometry(coo_)) {
      if (msat.valueAt(idx_) == 0)
        continue;

      m_ = mField.vectorAt(idx_);
    }
    else { // TODO: add proper (Neumann) BC
      m_ = m;
    }

    h += dot(normal, C) * dot(normal, w * w) * (m_ - m);
  }

  // MIXED DERIVATIVE
  const int3 c_xp_yp = mastergrid.wrap(coo + int3{+1, +1, 0});
  const int3 c_xp_ym = mastergrid.wrap(coo + int3{+1, -1, 0});
  const int3 c_xm_yp = mastergrid.wrap(coo + int3{-1, +1, 0});
  const int3 c_xm_ym = mastergrid.wrap(coo + int3{-1, -1, 0});


  real3 m_xp_yp = hField.cellInGeometry(c_xp_yp) ? mField.vectorAt(c_xp_yp) : real3{0, 0, 0};
  real3 m_xp_ym = hField.cellInGeometry(c_xp_ym) ? mField.vectorAt(c_xp_ym) : real3{0, 0, 0};
  real3 m_xm_yp = hField.cellInGeometry(c_xm_yp) ? mField.vectorAt(c_xm_yp) : real3{0, 0, 0};
  real3 m_xm_ym = hField.cellInGeometry(c_xm_ym) ? mField.vectorAt(c_xm_ym) : real3{0, 0, 0};

  real check = 1.0;
  if (m_xp_yp == real3{0, 0, 0}) { check = 0.0; }
  if (m_xp_ym == real3{0, 0, 0}) { check = 0.0; }
  if (m_xm_yp == real3{0, 0, 0}) { check = 0.0; }
  if (m_xm_ym == real3{0, 0, 0}) { check = 0.0; }

  real denom = w.x * w.y;
  // TODO: replace with higher order stencil once BC are included.
  //real3 dxy_m_second = (1.0/48.0) * (m_xp2_yp2 + m_xm2_ym2 - m_xm2_yp2 - m_xp2_ym2);

  h += C.z * check * (1.0/4.0) * ((m_xp_yp + m_xm_ym) - (m_xp_ym + m_xm_yp)) * denom;
  hField.setVectorInCell(idx, h / msat.valueAt(idx));
}


Field evalAtmExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3, real3{0, 0, 0});

  if (atmExchangeAssuredZero(magnet))
    return hField;
  real3 c = magnet->cellsize();
  real3 w = {1 / c.x, 1 / c.y, 1 / c.z};
  auto mag = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();

  auto host = magnet->hostMagnet();
  auto A1 = host->asATM()->A1.cu();
  auto A2 = host->asATM()->A2.cu();
  auto angle = host->asATM()->angle.cu();

  if (host->getSublatticeIndex(magnet) == 0)
    cudaLaunch(hField.grid().ncells(), k_atmExchangeField, hField.cu(),
                mag, A1, A2, angle, msat, magnet->world()->mastergrid(), w);
  else // Switch A1 and A2
    cudaLaunch(hField.grid().ncells(), k_atmExchangeField, hField.cu(),
                mag, A2, A1, angle, msat, magnet->world()->mastergrid(), w);
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