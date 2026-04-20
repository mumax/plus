#pragma once

#include "quantityevaluator.hpp"

class Altermagnet;
class Antiferromagnet;
class Ferromagnet;
class Field;
class NcAfm;

Field evalFMFullMag(const Ferromagnet*);
Field evalHMFullMag(const HostMagnet*);

FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet*);
AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet*);
ATM_FieldQuantity fullMagnetizationQuantity(const Altermagnet*);
NcAfm_FieldQuantity fullMagnetizationQuantity(const NcAfm*);