#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool externalFieldAssuredZero(const Ferromagnet*);
Field evalExternalField(const Ferromagnet*);
Field evalZeemanEnergyDensity(const Ferromagnet*);
real zeemanEnergy(const Ferromagnet*);

FM_FieldQuantity externalFieldQuantity(const Ferromagnet*);
FM_FieldQuantity zeemanEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity zeemanEnergyQuantity(const Ferromagnet*);
