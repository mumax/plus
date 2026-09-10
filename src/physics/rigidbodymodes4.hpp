#pragma once
#include "datatypes.hpp"

class Field;
class Magnet;

// ---------------------------------------------------------------------------
// Exact (non-linearized) rigid-body alignment via Horn's closed-form
// quaternion method (Horn, 1987, "Closed-form solution of absolute
// orientation using unit quaternions").
//
// Mathematically equivalent to the Kabsch/SVD approach in rigidbodymodes3.cu
// -- both extract the optimal rotation from the same cross-covariance matrix
// H -- but reformulated as: build a symmetric 4x4 matrix N from H, and take
// the eigenvector of N's largest eigenvalue as the optimal unit quaternion.
//
// This avoids the near-zero-singular-value special case that Kabsch-via-SVD
// needs (see rigidbodymodes3.cu's sigmaEpsilon fallback): there's no
// division by a singular value anywhere here, just an eigendecomposition of
// a well-posed symmetric matrix. The only remaining ambiguity is a
// *repeated* largest eigenvalue of N, which reflects the alignment problem
// itself being underdetermined (e.g. a mass distribution with no
// distinguishable rotation axis), not a numerical artifact of the method.
//
// Same caveats/intended use as rigidbodymodes3.cu: this is for one-off
// exact checks (e.g. a large built-in rotation from a dynamic-solver
// handoff), not a per-minimizer-step cleanup routine.
// ---------------------------------------------------------------------------

struct RigidBodyGeometry4 {
  real3 com0;      // reference (undeformed) mass-weighted center of mass
  real3 com0Unweighted;      // geometric centroid reference (no rho)
  real totalRho;   // total mass (sum of rho) in geometry
  real ncellsInGeometry;
  double S0[3][3];   // Sum w*(r0-com0)(r0-com0)^T -- reference shape covariance
  double S0Unweighted[3][3];  // unweighted reference shape covariance
};

struct QuatAlignResult {
  double q[4];      // optimal unit quaternion [q0 (scalar), q1, q2, q3]
  double R[3][3];   // rotation matrix equivalent to q
  real3 T;        // translation: x_current ~= R*r0 + T
  real3 com;      // current (deformed) mass-weighted center of mass
  double eigenGap;  // (largest - second largest) eigenvalue of N: near 0
                    // means the rotation axis is poorly determined by this
                    // mass/displacement configuration, not a method failure.
};

RigidBodyGeometry4 computeRigidBodyGeometry4(const Magnet* magnet);

QuatAlignResult computeQuaternionAlignment(const Field& u, const RigidBodyGeometry4& geom,
                                           const Magnet* magnet, bool unweighted = false);

// Subtracts the rigid (translation + finite rotation) component from a
// displacement field u, in place, using the quaternion alignment.
void removeRigidBodyModesQuaternion(Field& u, const RigidBodyGeometry4& geom,
                                    const Magnet* magnet, bool unweighted = false);

// Rotation angle (radians, in [0, pi]): angle = 2*acos(|q0|).
double quaternionRotationAngle(const QuatAlignResult& result);
