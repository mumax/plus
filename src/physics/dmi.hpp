/**
 * This header files contains the declaration of functions which compute the
 * effective field term and energy(density) related to the Dzyaloshinskii-
 * Moriya interaction (DMI).
 *
 * In mumax5, the DMI is defined by the following local energy density:
 *
 *   e = D_ijk [ m_j d_i(m_k) - m_k d_i(m_j) ]
 *
 * with:
 *   -  i,j,k summation indices over x, y, and z.
 *   -  DMI tensor D_ijk, which is assymetric on j and k (D_ijk = - D_ikj)
 *   -  spatial derivative d_i(..) along direction i
 */

#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

/** Return true if the DMI field and energy(density) are assured to be zero
 *  without evaluating the quantities.
 *
 *  DMI field and energy(density) are assured to be zero if the ferromagnet's
 *  DMI tensor or saturation magnetization (msat) can assured to be zero.
 */
bool dmiAssuredZero(const Ferromagnet*);

/** Evaluate the effective magnetic field related to DMI. */
Field evalDmiField(const Ferromagnet*);

/** Evaluate the DMI energy density field of a ferromagnet. */
Field evalDmiEnergyDensity(const Ferromagnet*);

/** Integrate the DMI energy density field over the ferromagnet. */
real evalDmiEnergy(const Ferromagnet*);

/** Construct FM_FieldQuantity around evalDmiField(const * Ferromagnet). */
FM_FieldQuantity dmiFieldQuantity(const Ferromagnet*);

/** Construct FM_FieldQuantity around evalEnergyDensity(const * Ferromagnet).*/
FM_FieldQuantity dmiEnergyDensityQuantity(const Ferromagnet*);

/** Construct FM_FieldQuantity around evalDmiEnergy(const * Ferromagnet).*/
FM_ScalarQuantity dmiEnergyQuantity(const Ferromagnet*);
