#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool bulkDmiAssuredZero(const Ferromagnet*);
Field evalBulkDmiField(const Ferromagnet*);
Field evalBulkDmiEnergyDensity(const Ferromagnet*);
real evalBulkDmiEnergy(const Ferromagnet*);

FM_FieldQuantity bulkDmiFieldQuantity(const Ferromagnet*);
FM_FieldQuantity bulkDmiEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity bulkDmiEnergyQuantity(const Ferromagnet*);
