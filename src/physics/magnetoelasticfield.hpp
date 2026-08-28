#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;


// Assure that the magnetoelastic field and force are 0
bool magnetoelasticAssuredZero(const Ferromagnet*);
bool dynamicMagnetoelasticAssuredZero(const Ferromagnet*);
bool rigidMagnetoelasticAssuredZero(const Ferromagnet*);

Field evalMagnetoelasticField(const Ferromagnet*);
Field evalMagnetoelasticEnergyDensity(const Ferromagnet*);
real evalMagnetoelasticEnergy(const Ferromagnet*);

// Magnetoelastic effective field due to effects of inverse magnetostriction
FM_FieldQuantity magnetoelasticFieldQuantity(const Ferromagnet*);
FM_FieldQuantity magnetoelasticEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity magnetoelasticEnergyQuantity(const Ferromagnet*);
