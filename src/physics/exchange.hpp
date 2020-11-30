#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool exchangeAssuredZero(const Ferromagnet*);
Field evalExchangeField(const Ferromagnet*);
Field evalExchangeEnergyDensity(const Ferromagnet*);
real evalExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity exchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity exchangeEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity exchangeEnergyQuantity(const Ferromagnet*);

// returns the maximal angle between exchange coupled cells
real evalMaxAngle(const Ferromagnet*);
FM_ScalarQuantity maxAngle(const Ferromagnet*);
