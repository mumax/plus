#include "linsolver.hpp"

#include <memory>
#include <utility>

#include "linsystem.hpp"
#include "vec.hpp"

LinSolver::LinSolver() : LinSolver(LinearSystem()) {}

LinSolver::LinSolver(LinearSystem system) : system_{std::move(system)} {
  stepper_ = Stepper::create(this, JACOBI);
  resetState();
}

void LinSolver::setSystem(LinearSystem newSystem) {
  system_ = std::move(newSystem);
  if (system_.nRows() != x_.size()) {
    resetState();
  }
}

void LinSolver::setMethod(Method method) {
  stepper_ = Stepper::create(this, method);
  stepper_->restart();
}

void LinSolver::setMethod(const std::string& methodname) {
  setMethod(getMethodByName(methodname));
}

void LinSolver::resetState() {
  x_ = GVec(system_.nRows());
  stepper_->restart();
}

GVec LinSolver::getState() const {
  return x_;
}

void LinSolver::setState(GVec newx) {
  if (newx.size() != system_.nRows())
    std::runtime_error(
        "The size of the input vector does not match the number of rows in the "
        "linear system.");
  x_ = std::move(newx);
  stepper_->restart();
}

GVec LinSolver::solve() {
  if (system_.empty())
    return GVec(0);

  int nstep = 0;
  while (residualMaxNorm() > tol) {
    if (nstep > max_iter && max_iter >= 0) {
      break;
    }

    stepper_->step();
    nstep++;
  }
  return x_;
}

LinSolver::Stepper* LinSolver::stepper() {
  return stepper_.get();
}

GVec LinSolver::residual() const {
  return system_.residual(x_);
}

double LinSolver::residualMaxNorm() const {
  return maxAbsValue(system_.residual(x_));
}

//--- IMPLEMENTATION OF STEPPERS -----------------------------------------------
// These implementations should only be used in this compilation unit, hence the
// anonymous namespace. If you implement a stepper, make sure to add the method
// to LinSolver::Method, in LinSolverStepper::create and in
// LinSolver::getMethodByName.
//------------------------------------------------------------------------------
namespace {

/**
 * Jacobi stepper to solve systems of linear equations
 * @see https://en.wikipedia.org/wiki/Jacobi_method
 */
class Jacobi : public LinSolver::Stepper {
 public:
  explicit Jacobi(LinSolver* parent) : LinSolver::Stepper(parent) {}
  Method method() const { return Method::JACOBI; }

  void step() {
    GVec& x = xRef();
    GVec r = system()->residual(x);  // r = b-Ax
    x = add(x, r);                   // x = x+r
  }
};

/**
 * Conjugate gradient stepper to solve systems of linear equations.
 * @see https://en.wikipedia.org/wiki/Conjugate_gradient_method
 */
class ConjugateGradient : public LinSolver::Stepper {
 public:
  explicit ConjugateGradient(LinSolver* parent) : LinSolver::Stepper(parent) {}
  Method method() const { return Method::CONJUGATEGRADIENT; }

  void step() {
    GVec& x = xRef();
    GVec Ap = system()->matrixmul(p);   // Ap  = A*p
    lsReal alpha = rr / dotSum(p, Ap);  // α   = rr/(p,Ap)
    x = add(1.0, x, alpha, p);          // x   = x + α*p
    r = add(1.0, r, -alpha, Ap);        // r   = r - α*Ap
    lsReal rrPrime = rr;                // rr' = rr
    rr = dotSum(r, r);                  // rr  = (r,r)
    lsReal beta = rr / rrPrime;         // β   = rr/rr'
    p = add(1.0, r, beta, p);           // p   = r + β*p
  }

  void restart() {
    GVec& x = xRef();
    r = system()->residual(x);  // r = b-A*x
    p = r;                      // p = r
    rr = dotSum(r, r);          // rr = (r,r)
  }

 private:
  lsReal rr;
  GVec p;
  GVec r;
};

/**
 * Minimal residual stepper to solve systems of linear equations.
 */
class MinimalResidual : public LinSolver::Stepper {
 public:
  explicit MinimalResidual(LinSolver* parent) : LinSolver::Stepper(parent) {}
  Method method() const { return Method::MINIMALRESIDUAL; }

  void step() {
    GVec& x = xRef();
    lsReal pp = dotSum(p, p);    // pp = (p,p)
    lsReal alpha = rr / pp;      // α  = rr/pp
    x = add(1.0, x, alpha, r);   // x  = x + α*r
    r = add(1.0, r, -alpha, p);  // r  = r - α*p
    p = system()->matrixmul(r);  // p  = A*r
    rr = dotSum(r, r);           // rr = (r,r)
  }

  void restart() {
    GVec& x = xRef();
    r = system()->residual(x);   // r  = b-A*x
    p = system()->matrixmul(r);  // p  = A*r
    rr = dotSum(r, r);           // rr = (r,r)
  }

 private:
  lsReal rr;
  GVec p;
  GVec r;
};

/**
 * Steepest descent stepper to solve systems of linear equations.
 * @see
 * https://www-m2.ma.tum.de/foswiki/pub/M2/Allgemeines/CSENumerikWS12/15_handout_iterative_III.pdf
 */
class SteepestDescent : public LinSolver::Stepper {
 public:
  explicit SteepestDescent(LinSolver* parent) : LinSolver::Stepper(parent) {}
  Method method() const { return Method::STEEPESTDESCENT; }

  void step() {
    GVec& x = xRef();
    lsReal pr = dotSum(p, r);    // pr = (p,r)
    lsReal alpha = rr / pr;      // α  = rr/pr
    x = add(1.0, x, alpha, r);   // x  = x + α*r
    r = add(1.0, r, -alpha, p);  // r  = r - α*p
    p = system()->matrixmul(r);  // p  = A*r
    rr = dotSum(r, r);           // rr = (r,r)
  }

  void restart() {
    GVec& x = xRef();
    r = system()->residual(x);   // r  = b-A*x
    p = system()->matrixmul(r);  // p  = A*r
    rr = dotSum(r, r);           // rr = (r,r)
  }

 private:
  lsReal rr;
  GVec p;
  GVec r;
};

}  // namespace

//--- LINSOLVER FACTORY METHOD -----------------------------------

std::unique_ptr<LinSolver::Stepper> LinSolver::Stepper::create(
    LinSolver* parent,
    Method method) {
  switch (method) {
    case Method::JACOBI:
      return std::make_unique<Jacobi>(parent);
    case Method::CONJUGATEGRADIENT:
      return std::make_unique<ConjugateGradient>(parent);
    case Method::MINIMALRESIDUAL:
      return std::make_unique<MinimalResidual>(parent);
    case Method::STEEPESTDESCENT:
      return std::make_unique<SteepestDescent>(parent);
    default:  // should never be reached
      throw std::invalid_argument("Linear system solver method number '"
                       + std::to_string(method) + "' does not exist");
  }
}

LinSolver::Method LinSolver::getMethodByName(const std::string& methodName) {
  if (methodName == "jacobi")
    return Method::JACOBI;
  if (methodName == "conjugategradient")
    return Method::CONJUGATEGRADIENT;
  if (methodName == "minimalresidual")
    return Method::MINIMALRESIDUAL;
  if (methodName == "steepestdescent")
    return Method::STEEPESTDESCENT;
  throw std::invalid_argument("Linear system solver method '" + methodName +
                              "' is not implemented");
}
