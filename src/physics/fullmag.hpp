#pragma once

#include "quantityevaluator.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;
class NcAfm;

Field evalFMFullMag(const Ferromagnet*);
Field evalHMFullMag(const HostMagnet*);

FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet*);
AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet*);
NcAfm_FieldQuantity fullMagnetizationQuantity(const NcAfm*);