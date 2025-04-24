#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

Field evalMagneticForceMicroscopy(const Ferromagnet*);
Field evalMagneticForceMicroscopyAFM(const Antiferromagnet*);

FM_FieldQuantity magneticForceMicroscopyQuantity(const Ferromagnet*);
AFM_FieldQuantity magneticForceMicroscopyAFMQuantity(const Antiferromagnet*);
