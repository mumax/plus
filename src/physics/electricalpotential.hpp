#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

bool electricalPotentialAssuredZero(const Ferromagnet*);
Field evalElectricalPotential(const Ferromagnet*);
FM_FieldQuantity electricalPotentialQuantity(const Ferromagnet*);
