#pragma once

#include <array>
#include "rigidbodymodes_common.hpp"

class Field;
class Magnet;

struct RigidModeMoments {
  double3 T;
  double3 omega;
};

struct RigidBodyGeometry {
  real3 com;
  real3 comUnweighted;    // geometric centroid (no rho)
  real totalRho;
  real ncellsInGeometry;
  double I[3][3];              
  double Iinv[3][3];           
  double IUnweighted[3][3];    
  double IinvUnweighted[3][3]; 
};

RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet);

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom,
                          const Magnet* magnet, bool unweighted = false);
