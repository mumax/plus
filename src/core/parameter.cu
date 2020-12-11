#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "system.hpp"

Parameter::Parameter(std::shared_ptr<const System> system, real value)
    : system_(system), field_(nullptr), uniformValue_(value) {}

Parameter::~Parameter() {
  if (field_)
    delete field_;
}

void Parameter::set(real value) {
  uniformValue_ = value;
  if (field_) {
    delete field_;
    field_ = nullptr;
  }
}

void Parameter::set(const Field& values) {
  field_ = new Field(values);
}

void Parameter::addTimeTerm(const time_function& term) {
  time_dep_terms.push_back(term);
}

bool Parameter::isUniform() const {
  return !field_;
}

bool Parameter::assuredZero() const {
  return isUniform() && uniformValue_ == 0.0;
}

int Parameter::ncomp() const {
  return 1;
}

std::shared_ptr<const System> Parameter::system() const {
  return system_;
}

Field Parameter::eval() const {
  Field p(system_, ncomp());
  if (field_) {
    p = *field_;
  } else {
    p.setUniformComponent(0, uniformValue_);
  }
  return p;
}

CuParameter Parameter::cu() const {
  return CuParameter(this);
}

VectorParameter::VectorParameter(std::shared_ptr<const System> system,
                                 real3 value)
    : system_(system), field_(nullptr), uniformValue_(value) {}

VectorParameter::~VectorParameter() {
  if (field_)
    delete field_;
}

void VectorParameter::set(real3 value) {
  uniformValue_ = value;
  if (field_)
    delete field_;
}

void VectorParameter::set(const Field& values) {
  field_ = new Field(values);
}

bool VectorParameter::isUniform() const {
  return !field_;
}

bool VectorParameter::assuredZero() const {
  return isUniform() && uniformValue_ == real3{0.0, 0.0, 0.0};
}

int VectorParameter::ncomp() const {
  return 3;
}

std::shared_ptr<const System> VectorParameter::system() const {
  return system_;
}

Field VectorParameter::eval() const {
  Field p(system(), ncomp());
  if (field_) {
    p = *field_;
  } else {
    p.setUniformComponent(0, uniformValue_.x);
    p.setUniformComponent(1, uniformValue_.y);
    p.setUniformComponent(2, uniformValue_.z);
  }
  return p;
}

CuVectorParameter VectorParameter::cu() const {
  return CuVectorParameter(this);
}
