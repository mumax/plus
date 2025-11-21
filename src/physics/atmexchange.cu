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
  if (!magnet->hostMagnet()) { return true; }
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
                                const CuField m1Field,
                                const CuField m2Field,
                                const CuParameter aex,
                                const CuParameter afmex_nn,
                                const CuInterParameter interExch,
                                const CuInterParameter scaleExch,
                                const CuParameter msat,
                                const CuParameter msat2,
                                const Grid mastergrid,
                                const CuDmiTensor dmiTensor,
                                bool openBC) {

  // INSERT IMPLEMENTATION HERE
                                }

Field evalAtmExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3, real3{0, 0, 0});

  if (atmExchangeAssuredZero(magnet))
    return hField;

  auto aex = magnet->aex.cu();
  auto dmiTensor = magnet->dmiTensor.cu();
  auto BC = magnet->enableOpenBC;
  auto mag = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();

  auto host = magnet->hostMagnet();
  auto atmex_ani = host->asATM()->atmex_ani.cu();
  auto inter = host->interAfmExchNN.cu();
  auto scale = host->scaleAfmExchNN.cu();

  for (auto sub : host->getOtherSublattices(magnet)) {
    // Accumulate seperate sublattice contributions
    auto mag2 = sub->magnetization()->field().cu();
    auto msat2 = sub->msat.cu();
    cudaLaunch(hField.grid().ncells(), k_atmExchangeField, hField.cu(),
              mag, mag2, aex, atmex_ani, inter, scale, msat, msat2,
              magnet->world()->mastergrid(), dmiTensor, BC);
  }
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