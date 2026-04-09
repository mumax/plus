#include "altermagnet.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
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

  return false;
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
  const real3 w2 = w * w;
  const real3 m = mField.vectorAt(idx);
  const real a1 = A1.valueAt(idx);
  const real a2 = A2.valueAt(idx);

  // CALCULATE PREFACTORS
  // TODO: add inter-values of ai
  real c = cos(angle.valueAt(idx));
  real s = sin(angle.valueAt(idx));
  real c2 = c * c;
  real s2 = s * s;
  real Cxy = 2 * c * s * (a1 - a2);

  real3 h{0, 0, 0};

  // SECOND ORDER DERIVATIVES
  // TODO: add proper (Neumann) BC once mixed derivative formulation is clear
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0}, int3{0, 1, 0}}) {
    const int3 coo_ = mastergrid.wrap(coo + rel_coo);
    const int idx_ = grid.coord2index(coo_);

    real3 m_;
    real a1_;
    real a2_;
    int3 normal = rel_coo * rel_coo;

    if(hField.cellInGeometry(coo_)) {
      if (msat.valueAt(idx_) == 0)
        continue;

      m_ = mField.vectorAt(idx_);
      a1_ = A1.valueAt(idx_);
      a2_ = A2.valueAt(idx_);
    }
    else {
      m_ = m;
      a1_ = a1;
      a2_ = a2;
    }

    real aex_x = harmonicMean(a1, a1_);
    real aex_y = harmonicMean(a2, a2_);
    real Aex;

    if (rel_coo.x != 0) { Aex = aex_x * c2 + aex_y * s2; }
    else { Aex = aex_x * s2 + aex_y * c2; }

    h += Aex * dot(normal, w2) * (m_ - m);
  }

  // MIXED DERIVATIVE
  if (Cxy == 0) {
    hField.setVectorInCell(idx, h / msat.valueAt(idx));
    return;
  }

  // TODO: add proper (Neumann) BC
  const int3 rel_coos[4] = { {+1, +1, 0}, {+1, -1, 0}, {-1, +1, 0}, {-1, -1, 0} };
  int3 coos[4];
  bool inBulk = true;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    coos[i] = mastergrid.wrap(coo + rel_coos[i]);

    inBulk &= hField.cellInGeometry(coos[i]);
    inBulk &= (msat.valueAt(coos[i]) != 0);
  }

  if (!inBulk) {
    hField.setVectorInCell(idx, h / msat.valueAt(idx));
    return;
  }

  const real3 m_xp_yp = mField.vectorAt(coos[0]);
  const real3 m_xp_ym = mField.vectorAt(coos[1]);
  const real3 m_xm_yp = mField.vectorAt(coos[2]);
  const real3 m_xm_ym = mField.vectorAt(coos[3]);

  h += Cxy * (1.0/4.0) * ((m_xp_yp + m_xm_ym) - (m_xp_ym + m_xm_yp)) * w.x * w.y;
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