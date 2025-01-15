#pragma once

#include <memory>

#include "datatypes.hpp"

class Variable;
class FieldQuantity;
class Grid;
class System;

class DynamicEquation {
 private:
  /** Max error for adaptive time stepping, pointing to set value of a magnet.
   *  If not set, the default is 1e-5.
   */
  const real* maxError_;

 public:
  DynamicEquation(const Variable* x,
                  std::shared_ptr<FieldQuantity> rhs,
                  std::shared_ptr<FieldQuantity> noiseTerm = nullptr,
                  const real* maxError = nullptr);
  DynamicEquation(const Variable* x,
                  std::shared_ptr<FieldQuantity> rhs,
                  const real* maxError);  // option without thermal noise

  const Variable* x;
  std::shared_ptr<FieldQuantity> rhs;
  std::shared_ptr<FieldQuantity> noiseTerm;

  int ncomp() const;
  Grid grid() const;
  std::shared_ptr<const System> system() const;
  real maxError() const;
};
