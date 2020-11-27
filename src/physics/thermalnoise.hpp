#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool thermalNoiseAssuredZero(const Ferromagnet* magnet);

Field evalThermalNoise(const Ferromagnet*);

FM_FieldQuantity thermalNoiseQuantity(const Ferromagnet*);
