#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

Field evalTorque(const Ferromagnet*);
Field evalLlgTorque(const Ferromagnet*);
Field evalRelaxTorque(const Ferromagnet*);

FM_FieldQuantity torqueQuantity(const Ferromagnet*);
FM_FieldQuantity llgTorqueQuantity(const Ferromagnet*);
FM_FieldQuantity relaxTorqueQuantity(const Ferromagnet*);
