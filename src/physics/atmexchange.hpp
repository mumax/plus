#pragma once

#include "quantityevaluator.hpp"

class Altermagnet;
class Ferromagnet;
class Field;

// Anisotropic exchange contribution

bool atmExchangeAssuredZero(const Ferromagnet*);

// Evaluate field
Field evalAtmExchangeField(const Ferromagnet*);
// Evaluate energy density
Field evalAtmExchangeEnergyDensity(const Ferromagnet*);
// Evaluate energy
real evalAtmExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity atmExchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity atmExchangeEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity atmExchangeEnergyQuantity(const Ferromagnet*);