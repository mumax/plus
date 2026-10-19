#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "gpubuffer.hpp"
#include "linsystem.hpp"
#include "vec.hpp"

/** Solves a linear solver Ax=b iteratively. */
class LinSolver {
 public:
  class Stepper;
  enum Method { JACOBI, CONJUGATEGRADIENT, MINIMALRESIDUAL, STEEPESTDESCENT };
  static Method getMethodByName(const std::string&);

 public:
  LinSolver();
  explicit LinSolver(LinearSystem system);

  /** Sets the system of linear equations to be solved. resetState is
   * called if and only if the size of the internal state does no longer match
   * the number of rows of the system.
   */
  void setSystem(LinearSystem newSystem);

  Stepper* stepper();

  void setMethod(Method);
  void setMethod(const std::string& methodName);

  GVec getState() const;    /** Get x. */
  void setState(GVec newx); /** Set x to newx and reset stepper. */
  void resetState();        /** Set x to zero and reset stepper. */

  GVec residual() const;          /** Return residual b-Ax. */
  double residualMaxNorm() const; /** Return max absolute value of b-Ax. */

  /** Use the stepper to let x converge to the solution of the linear system.
   * The iteration stops when the max norm of the residual (residualMaxNorm())
   * is smaller than the tollerance (tol) or when the maximum number of
   * iterations (max_iter) is reached.
   */
  GVec solve();

 public:
  int max_iter = -1; /** Maximum iterations allowed when calling solve(). */
  double tol = 1e-5; /** Convergence tollerance on max norm of the residual. */

 private:
  std::unique_ptr<Stepper> stepper_;
  LinearSystem system_;
  GVec x_; /** The intermediate internal solution of the system. */
};

class LinSolver::Stepper {
 public:
  using Method = LinSolver::Method;
  static std::unique_ptr<Stepper> create(LinSolver* parent, Method);

 public:
  explicit Stepper(LinSolver* parent) : parent_(parent) {}
  virtual Method method() const = 0;
  virtual void step() = 0;
  virtual void restart() {}

 protected:
  LinearSystem* system() const { return &(parent_->system_); }
  GVec& xRef() const { return parent_->x_; }

 private:
  LinSolver* parent_;
};
