#pragma once

#include "quantityevaluator.hpp"

class Altermagnet;
class Antiferromagnet;
class Field;

Field evalNeelvector(const HostMagnet*);

AFM_FieldQuantity neelVectorQuantity(const Antiferromagnet*);
ATM_FieldQuantity neelVectorQuantity(const Altermagnet*);