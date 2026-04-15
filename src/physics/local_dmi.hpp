#pragma once

#include "quantityevaluator.hpp"

bool homoDmiAssuredZero(const Ferromagnet*);

Field evalHomoDmiField(const Ferromagnet*);
Field evalHomoDmiEnergyDensity(const Ferromagnet*);
real evalHomoDmiEnergy(const Ferromagnet*);

FM_FieldQuantity homoDmiFieldQuantity(const Ferromagnet*);
FM_FieldQuantity homoDmiEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity homoDmiEnergyQuantity(const Ferromagnet*);