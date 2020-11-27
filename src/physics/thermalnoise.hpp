#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

Field evalThermalNoise(const Ferromagnet*);

FM_FieldQuantity thermalNoiseQuantity(const Ferromagnet*);
