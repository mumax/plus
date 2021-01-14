#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "parameter.hpp"

Parameter::Parameter(std::shared_ptr<const System> system, real value)
    : system_(system), staticField_(nullptr), uniformValue_(value) {}

Parameter::~Parameter() {
    if (staticField_)
        delete staticField_;
}

void Parameter::set(real value) {
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
}

void Parameter::set(const Field& values) {
   staticField_ = new Field(values);
}

bool Parameter::isUniform() const {
  return !staticField_;
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
  auto t = system_->world()->time();
  Field staticField(system_, ncomp());
  Field dynamicField(system_, ncomp());

  dynamicField.setUniformComponent(0, 0);
  evalTimeDependentTerms(t, dynamicField);

  if (staticField_) {
      staticField = *staticField_;
  }
  else {
      staticField.setUniformComponent(0, uniformValue_);
  }

  staticField += dynamicField;

  return staticField;
}

CuParameter Parameter::cu() const {
    auto t = system_->world()->time();

    dynamicField_.reset(new Field(system_, ncomp()));
    dynamicField_->setUniformComponent(0, 0);

    evalTimeDependentTerms(t, *dynamicField_);
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
