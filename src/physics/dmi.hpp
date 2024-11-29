/**
 * This header files contains the declaration of functions which compute the
 * effective field term and energy(density) related to the Dzyaloshinskii-
 * Moriya interaction (DMI).
 *
 * In mumax⁺, the DMI is defined by the following local energy density:
 *
 *   e = D_ijk [ m_j d_i(m_k) - m_k d_i(m_j) ]
 *
 * with:
 *   -  i,j,k summation indices over x, y, and z.
 *   -  DMI tensor D_ijk, which is assymetric on j and k (D_ijk = - D_ikj)
 *   -  spatial derivative d_i(..) along direction i
 *
 * Neumann boundary conditions are assumed, unless specified otherwise (i.e. open
 * boundaries).
 */

#pragma once

#include "dmitensor.hpp"
#include "quantityevaluator.hpp"

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

//------------------------- HELPER FUNCTION ------------------------------
//     Device function defined here to be used both here and elsewhere.
//------------------------------------------------------------------------

__device__ static inline real3 getGamma(const CuDmiTensor dmiTensor,
                                        const int idx, int3 n, real3 m) {
  // returns the DMI field at the boundary
  real Dxxz = dmiTensor.xxz.valueAt(idx);
  real Dxxy = dmiTensor.xxy.valueAt(idx);
  real Dxyz = dmiTensor.xyz.valueAt(idx);
  real Dyxz = dmiTensor.yxz.valueAt(idx);
  real Dyxy = dmiTensor.yxy.valueAt(idx);
  real Dyyz = dmiTensor.yyz.valueAt(idx);
  real Dzxz = dmiTensor.zxz.valueAt(idx);
  real Dzxy = dmiTensor.zxy.valueAt(idx);
  real Dzyz = dmiTensor.zyz.valueAt(idx);
  return real3{
        -Dxxy*n.x*m.y - Dxxz*n.x*m.z - Dyxz*n.y*m.z - Dzxy*n.z*m.y - Dyxy*n.y*m.y - Dzxz*n.z*m.z,
         Dxxy*n.x*m.x - Dzyz*n.z*m.z + Dyxy*n.y*m.x - Dxyz*n.x*m.z + Dzxy*n.z*m.x - Dyyz*n.y*m.z,
         Dxxz*n.x*m.x + Dyyz*n.y*m.y + Dxyz*n.x*m.y + Dyxz*n.y*m.x + Dzxz*n.z*m.x + Dzyz*n.z*m.y};
}