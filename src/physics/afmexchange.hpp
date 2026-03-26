#pragma once

#include "quantityevaluator.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

// The homogeneous and inhomogeneous contributions to AFM exchange is split up.
// The homogeneous contribution (considering afmex_cell) corresponds to AFM
// exchange at a single site.
// The inhomogeneous contribution (considering afmex_nn) corresponds to AFM
// exchange between NN cells.

bool inHomoAfmExchangeAssuredZero(const Ferromagnet*);
bool homoAfmExchangeAssuredZero(const Ferromagnet*);

// Evaluate field
Field evalInHomogeneousAfmExchangeField(const Ferromagnet*);
Field evalHomogeneousAfmExchangeField(const Ferromagnet*);
// Evaluate energy density
Field evalInHomoAfmExchangeEnergyDensity(const Ferromagnet*);
Field evalHomoAfmExchangeEnergyDensity(const Ferromagnet*);
// Evaluate energy
real evalInHomoAfmExchangeEnergy(const Ferromagnet*);
real evalHomoAfmExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity inHomoAfmExchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity homoAfmExchangeFieldQuantity(const Ferromagnet*);

FM_FieldQuantity inHomoAfmExchangeEnergyDensityQuantity(const Ferromagnet*);
FM_FieldQuantity homoAfmExchangeEnergyDensityQuantity(const Ferromagnet*);

FM_ScalarQuantity inHomoAfmExchangeEnergyQuantity(const Ferromagnet*);
FM_ScalarQuantity homoAfmExchangeEnergyQuantity(const Ferromagnet*);

////////////////////////////////////////////////////////////////////////////////////
// TODO: some day, someone should put the angle calculation in a different file.
// returns the deviation from the optimal angle (180°) between magnetization
// vectors in the same cell which are coupled by the intracell exchange interaction.
Field evalAngleField(const HostMagnet*);
// The maximal deviation from 180*.
real evalMaxAngle(const HostMagnet*);

AFM_FieldQuantity angleFieldQuantity(const Antiferromagnet*);
ATM_FieldQuantity angleFieldQuantity(const Altermagnet*);
AFM_ScalarQuantity maxAngle(const Antiferromagnet*);
ATM_ScalarQuantity maxAngle(const Altermagnet*);




