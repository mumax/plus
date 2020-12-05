#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool demagFieldAssuredZero(const Ferromagnet*);
Field evalDemagField(const Ferromagnet*);
Field evalDemagEnergyDensity(const Ferromagnet*);
real evalDemagEnergy(const Ferromagnet*);

FM_FieldQuantity demagFieldQuantity(const Ferromagnet*);
FM_FieldQuantity demagEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity demagEnergyQuantity(const Ferromagnet*);
