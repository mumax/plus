#include "dynamicequation.hpp"

#include <exception>
#include <memory>

#include "fieldquantity.hpp"
#include "system.hpp"
#include "variable.hpp"

DynamicEquation::DynamicEquation(const Variable* x,
                                 std::shared_ptr<FieldQuantity> rhs,
                                 std::shared_ptr<FieldQuantity> noiseTerm)
    : x(x), rhs(rhs), noiseTerm(noiseTerm) {
  if (x->system() != rhs->system()) {
    throw std::runtime_error(
        "The variable and the r.h.s. of a dynamic equation should have the "
        "same underlying system");
  }
  if (x->ncomp() != rhs->ncomp()) {
    throw std::runtime_error(
        "The variable and the r.h.s. of a dynamic equation should have the "
        "same number of components");
  }

  if (noiseTerm) {
    if (x->system() != noiseTerm->system()) {
      throw std::runtime_error(
          "The variable and the noise term of a dynamic equation should have "
          "the same underlying system");
    }
    if (x->ncomp() != noiseTerm->ncomp()) {
      throw std::runtime_error(
          "The variable and the noise term of a dynamic equation should have "
          "the same number of components");
    }
  }
}

int DynamicEquation::ncomp() const {
  return x->ncomp();
}

Grid DynamicEquation::grid() const {
  return system()->grid();
}

std::shared_ptr<const System> DynamicEquation::system() const {
  return x->system();
}
