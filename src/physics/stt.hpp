#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool spinTransferTorqueAssuredZero(const Ferromagnet*);

Field evalSpinTransferTorque(const Ferromagnet*);

FM_FieldQuantity spinTransferTorqueQuantity(const Ferromagnet*);
