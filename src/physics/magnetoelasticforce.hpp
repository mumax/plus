#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;


Field evalMagnetoelasticForce(const Ferromagnet*);

// Magnetoelastic body force due to the magnetostriction effect.
FM_FieldQuantity magnetoelasticForceQuantity(const Ferromagnet*);
