#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

// Returns the energy density of an effective field term.
//   edens = - prefactor * Msat * dot(m,h)
// The prefactor depends on the origin of the effective field term
Field evalEnergyDensity(const Ferromagnet*, const Field&, real prefactor);

Field evalTotalEnergyDensity(const Ferromagnet*);
real evalTotalSublatticeEnergy(const Ferromagnet*, const bool sub2);
real evalTotalEnergy(const Ferromagnet*, const bool sub2);

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity totalSublatticeEnergyQuantity(const Ferromagnet* magnet, const bool sub2);
FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet*, const bool sub2);
