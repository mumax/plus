#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool interfacialDmiAssuredZero(const Ferromagnet*);
Field evalInterfacialDmiField(const Ferromagnet*);
Field evalInterfacialDmiEnergyDensity(const Ferromagnet*);
real evalInterfacialDmiEnergy(const Ferromagnet*);

FM_FieldQuantity interfacialDmiFieldQuantity(const Ferromagnet*);
FM_FieldQuantity interfacialDmiEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity interfacialDmiEnergyQuantity(const Ferromagnet*);
