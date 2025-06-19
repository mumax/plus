#pragma once

#include "parameter.hpp"
#include "system.hpp"

struct CuBoundaryTraction;

/** BoundaryTraction holds the external traction applied on the 6 faces of each cell. */
struct BoundaryTraction {
  VectorParameter posXside;
  VectorParameter negXside;
  VectorParameter posYside;
  VectorParameter negYside;
  VectorParameter posZside;
  VectorParameter negZside;

  /** Construct the boundary traction for a given system */
  explicit BoundaryTraction(std::shared_ptr<const System> system, std::string name = "");

  /** Return CuBoundaryTraction */
  CuBoundaryTraction cu() const;

  /** Returns true if all 6 VectorParameters are equal to zero. */
  bool assuredZero() const;
};


struct CuBoundaryTraction {
  CuVectorParameter posXside;
  CuVectorParameter negXside;
  CuVectorParameter posYside;
  CuVectorParameter negYside;
  CuVectorParameter posZside;
  CuVectorParameter negZside;

  __device__ const CuVectorParameter& getSide(int orientation, int sense) const;
};


__device__ inline const CuVectorParameter& CuBoundaryTraction::getSide(int orientation, int sense) const {
  // no safety measures
  if (orientation == 0) return (sense == 1) ? posXside : negXside;
  if (orientation == 1) return (sense == 1) ? posYside : negYside;
  return (sense == 1) ? posZside : negZside;
}
