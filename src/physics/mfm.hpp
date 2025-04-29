#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Antiferromagnet;
class Field;

Field evalMagneticForceMicroscopy(const Ferromagnet*);
Field evalMagneticForceMicroscopyAFM(const Antiferromagnet*);

FM_FieldQuantity magneticForceMicroscopyQuantity(const Ferromagnet*);
AFM_FieldQuantity magneticForceMicroscopyQuantity(const Antiferromagnet*);
