#include "altermagnet.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "dmi.hpp" // used for inter-regional exchange
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
  if ((magnet->hostMagnet()->asATM()->alterex_1.assuredZero() &&
       magnet->hostMagnet()->asATM()->alterex_2.assuredZero() &&
       magnet->hostMagnet()->asATM()->interAlterex_1.assuredZero() &&
       magnet->hostMagnet()->asATM()->interAlterex_2.assuredZero()) ||
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
                                const real3 w,
                                const CuInterParameter interEx1,
                                const CuInterParameter scaleEx1,
                                const CuInterParameter interEx2,
                                const CuInterParameter scaleEx2) {
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
  const real3 w2 = w * w;
  const real3 m = mField.vectorAt(idx);
  const real a1 = A1.valueAt(idx);
  const real a2 = A2.valueAt(idx);

  // CALCULATE PREFACTORS
  real c = cos(angle.valueAt(idx));
  real s = sin(angle.valueAt(idx));
  real c2 = c * c;
  real s2 = s * s;
  real cs2 = 2 * c * s;

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

    real inter1 = 0, inter2 = 0;
    real scale1 = 1, scale2 = 1;
    real Aex;

    if(hField.cellInGeometry(coo_)) {
      if (msat.valueAt(idx_) == 0)
        continue;

      m_ = mField.vectorAt(idx_);
      a1_ = A1.valueAt(idx_);
      a2_ = A2.valueAt(idx_);
      unsigned int ridx = system.getRegionIdx(idx);
      unsigned int ridx_ = system.getRegionIdx(idx_);

      if (ridx != ridx_) {
        scale1 = scaleEx1.valueBetween(ridx, ridx_);
        inter1 = interEx1.valueBetween(ridx, ridx_);
        scale2 = scaleEx2.valueBetween(ridx, ridx_);
        inter2 = interEx2.valueBetween(ridx, ridx_);
      }
    }
    else {
      m_ = m;
      a1_ = a1;
      a2_ = a2;
    }

    real aex_1 = getExchangeStiffness(inter1, scale1, a1, a1_);
    real aex_2 = getExchangeStiffness(inter2, scale2, a2, a2_);

    if (rel_coo.x != 0) { Aex = aex_1 * c2 + aex_2 * s2; }
    else { Aex = aex_1 * s2 + aex_2 * c2; }
    h += Aex * dot(normal, w2) * (m_ - m);
  }

  // MIXED DERIVATIVE
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

  unsigned int ridx = system.getRegionIdx(idx);
  real inter1 = 0, inter2 = 0;
  real scale1 = 1, scale2 = 1;
  real Aex[4];

#pragma unroll
  for (int i = 0; i < 4; i++) {
    unsigned int ridx_ = system.getRegionIdx(coos[i]);

    if (ridx != ridx) {
      scale1 = scaleEx1.valueBetween(ridx, ridx_);
      inter1 = interEx1.valueBetween(ridx, ridx_);
      scale2 = scaleEx2.valueBetween(ridx, ridx_);
      inter2 = interEx2.valueBetween(ridx, ridx_);
    }
    real aex_1 = getExchangeStiffness(inter1, scale1, a1, A1.valueAt(coos[i]));
    real aex_2 = getExchangeStiffness(inter2, scale2, a2, A2.valueAt(coos[i]));
    Aex[i] = cs2 * (aex_1 - aex_2);
  }

  const real3 m_xp_yp = mField.vectorAt(coos[0]);
  const real3 m_xp_ym = mField.vectorAt(coos[1]);
  const real3 m_xm_yp = mField.vectorAt(coos[2]);
  const real3 m_xm_ym = mField.vectorAt(coos[3]);

  h += (1.0/4.0) * ((Aex[0] * m_xp_yp + Aex[3] * m_xm_ym)
                  - (Aex[1] * m_xp_ym + Aex[2] * m_xm_yp)) * w.x * w.y;
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

  auto host = magnet->hostMagnet()->asATM();
  auto A1 = host->alterex_1.cu();
  auto A2 = host->alterex_2.cu();
  auto angle = host->alterex_angle.cu();
  auto inter1 = host->interAlterex_1.cu();
  auto scale1 = host->scaleAlterex_1.cu();
  auto inter2 = host->interAlterex_2.cu();
  auto scale2 = host->scaleAlterex_2.cu();

  if (host->getSublatticeIndex(magnet) == 0)
    cudaLaunch(hField.grid().ncells(), k_atmExchangeField, hField.cu(),
                mag, A1, A2, angle, msat, magnet->world()->mastergrid(),
                w, inter1, scale1, inter2, scale2);
  else // Switch A1 and A2
    cudaLaunch(hField.grid().ncells(), k_atmExchangeField, hField.cu(),
                mag, A2, A1, angle, msat, magnet->world()->mastergrid(),
                w, inter2, scale2, inter1, scale1);
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