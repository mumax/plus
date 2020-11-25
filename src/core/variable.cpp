#include "variable.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include "field.hpp"
#include "fieldops.hpp"
#include "system.hpp"

Variable::Variable(std::string name,
                   std::string unit,
                   std::shared_ptr<const System> system,
                   int ncomp)
    : name_(name), unit_(unit) {
  field_ = new Field(system, ncomp);
}

Variable::~Variable() {
  delete field_;
}

int Variable::ncomp() const {
  return field_->ncomp();
}

std::shared_ptr<const System> Variable::system() const {
  return field_->system();
}

std::string Variable::name() const {
  return name_;
}

std::string Variable::unit() const {
  return unit_;
}

Field Variable::eval() const {
  return field_->eval();
}

const Field& Variable::field() const {
  return *field_;
}

void Variable::set(const Field& src) const {
  if (src.system() != field_->system()) {
    throw std::runtime_error(
        "Can not set the variable because the given field variable is defined "
        "on another system.");
  }
  *field_ = src;
}

void Variable::set(real value) const {
  if (ncomp() != 1)
    throw std::runtime_error("Variable has " + std::to_string(ncomp()) +
                             "components instead of 1");
  field_->setUniformComponent(0, value);
}

void Variable::set(real3 value) const {
  if (ncomp() != 3)
    throw std::runtime_error("Variable has " + std::to_string(ncomp()) +
                             "components instead of 3");
  field_->setUniformComponent(0, value.x);
  field_->setUniformComponent(1, value.y);
  field_->setUniformComponent(2, value.z);
}

NormalizedVariable::NormalizedVariable(std::string name,
                                       std::string unit,
                                       std::shared_ptr<const System> system,
                                       int ncomp)
    : Variable(name, unit, system, ncomp) {}

void NormalizedVariable::set(const Field& src) const {
  // TODO: check if this is possible without the extra copy
  Variable::set(normalized(src));
}

void NormalizedVariable::set(real value) const {
  Variable::set(1);
}

void NormalizedVariable::set(real3 value) const {
  Variable::set(normalized(value));
}
