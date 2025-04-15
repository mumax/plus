#pragma once

#include "quantityevaluator.hpp"

class Antiferromagnet;
class NCAFM;
class Field;


bool homoDmiAssuredZero(const Ferromagnet*);

Field evalHomoDmiField(const Ferromagnet*);

FM_FieldQuantity homoDmiFieldQuantity(const Ferromagnet*);
FM_FieldQuantity homoDmiEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity homoDmiEnergyQuantity(const Ferromagnet*);