#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


bool kineticEnergyAssuredZero(const Magnet*);

Field evalKineticEnergyDensity(const Magnet*);
Field evalElasticEnergyDensity(const Magnet*);

real evalKineticEnergy(const Magnet*);
real evalElasticEnergy(const Magnet*);

// Elastic kinetic energy density
M_FieldQuantity kineticEnergyDensityQuantity(const Magnet*);
// Elastic kinetic energy
M_ScalarQuantity kineticEnergyQuantity(const Magnet*);

// Elastic potential energy density
M_FieldQuantity elasticEnergyDensityQuantity(const Magnet*);
// Elastic potential energy
M_ScalarQuantity elasticEnergyQuantity(const Magnet*);
