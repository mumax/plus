#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool anisotropyAssuredZero(const Ferromagnet*);
Field evalAnisotropyField(const Ferromagnet*);
Field evalAnisotropyEnergyDensity(const Ferromagnet*);
real evalAnisotropyEnergy(const Ferromagnet*);

FM_FieldQuantity anisotropyFieldQuantity(const Ferromagnet*);
FM_FieldQuantity anisotropyEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity anisotropyEnergyQuantity(const Ferromagnet*);
