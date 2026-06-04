#pragma once

#include "quantityevaluator.hpp"

// returns the deviation from the optimal angle (120°) between magnetization
// vectors in the same cell which are coupled by the intracell exchange interaction.
Field evalAngleField(const NcAfm*);
// The maximal angle deviation from 120° between two given sublattices.
real evalMaxAngle(const Ferromagnet*, const Ferromagnet*);

NcAfm_FieldQuantity angleFieldQuantity(const NcAfm*);