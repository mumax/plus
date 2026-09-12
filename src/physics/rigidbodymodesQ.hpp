#pragma once
#include "datatypes.hpp"

class Field;
class Magnet;

struct RigidBodyGeomQ {
  real3 com0;      // reference (undeformed) mass-weighted center of mass
  real3 com0Unweighted;      // reference geometric centroid (no rho)
  real totalRho;   
  real ncellsInGeometry;
  double S0[3][3];   // Σ_i w_i*(r0_i-com0)(r0_i-com0)^T -- reference shape covariance
  double S0Unweighted[3][3];  // unweighted reference shape covariance
};

struct QuatAlignResult {
  double q[4];      
  double R[3][3];   
  real3 T;        
  real3 com;      
  double eigenGap;  
};

RigidBodyGeomQ computeRigidBodyGeometryQ(const Magnet* magnet);

QuatAlignResult computeQuaternionAlignment(const Field& u, const RigidBodyGeomQ& geom,
                                           const Magnet* magnet, bool unweighted = false);

void removeRigidBodyModesQ(Field& u, const RigidBodyGeomQ& geom,
                                    const Magnet* magnet, bool unweighted = false);

double quaternionRotationAngle(const QuatAlignResult& result);
