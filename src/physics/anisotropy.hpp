#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

bool unianisotropyAssuredZero(const Ferromagnet*);
bool cubicanisotropyAssuredZero(const Ferromagnet*);
bool anisotropyAssuredZero(const Ferromagnet*);
Field evalAnisotropyField(const Ferromagnet*);
Field evalAnisotropyEnergyDensity(const Ferromagnet*);
real evalAnisotropyEnergy(const Ferromagnet*);

FM_FieldQuantity anisotropyFieldQuantity(const Ferromagnet*);
FM_FieldQuantity anisotropyEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity anisotropyEnergyQuantity(const Ferromagnet*);
FM_ScalarQuantity anisotropyEnergyQuantity_sub2(const Ferromagnet*);
