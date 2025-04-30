#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Antiferromagnet;
class Field;

Field evalMagneticForceMicroscopy(const Magnet*);

FM_FieldQuantity magneticForceMicroscopyQuantity(const Ferromagnet*);
AFM_FieldQuantity magneticForceMicroscopyQuantity(const Antiferromagnet*);
