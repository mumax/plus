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
  return !isDynamic() && isUniform() && uniformValue_ == 0.0;
}

int Parameter::ncomp() const {
  return 1;
}

std::shared_ptr<const System> Parameter::system() const {
  return system_;
}

Field Parameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
      staticField = *staticField_;
  }
  else {
      // can we safely skip n = 0 here???, can we move this function in a parent class???
      // e.g. Scalar Parameter instead of Parameter, and create Static Parameter or just Parameter???
      staticField.setUniformComponent(uniformValue_);
  }

  if (isDynamic()) {
      auto t = system_->world()->time();
      Field dynamicField(system_, ncomp());

      dynamicField.makeZero();
      evalTimeDependentTerms(t, dynamicField);

      staticField += dynamicField;
  }

  return staticField;
}

CuParameter Parameter::cu() const {
    if (isDynamic()) {
        auto t = system_->world()->time();
        dynamicField_.reset(new Field(system_, ncomp()));
        dynamicField_->makeZero();

        evalTimeDependentTerms(t, *dynamicField_);
    }

  return CuParameter(this);
}

VectorParameter::VectorParameter(std::shared_ptr<const System> system,
                                 real3 value)
    : system_(system), staticField_(nullptr), uniformValue_(value) {}

VectorParameter::~VectorParameter() {
  if (staticField_)
    delete staticField_;
}

void VectorParameter::set(real3 value) {
  uniformValue_ = value;
  if (staticField_)
    delete staticField_;
}

void VectorParameter::set(const Field& values) {
  staticField_ = new Field(values);
}

bool VectorParameter::isUniform() const {
  return !staticField_;
}

bool VectorParameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == real3{0.0, 0.0, 0.0};
}

int VectorParameter::ncomp() const {
  return 3;
}

std::shared_ptr<const System> VectorParameter::system() const {
  return system_;
}

Field VectorParameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
      staticField = *staticField_;
  }
  else {
      staticField.setUniformComponent(uniformValue_);
  }

  if (isDynamic()) {
      auto t = system_->world()->time();
      Field dynamicField(system_, ncomp());

      dynamicField.makeZero();
      evalTimeDependentTerms(t, dynamicField);

      staticField += dynamicField;
  }

  return staticField;
}

CuVectorParameter VectorParameter::cu() const {
    if (isDynamic()) {
        auto t = system_->world()->time();
        dynamicField_.reset(new Field(system_, ncomp()));
        dynamicField_->makeZero();

        evalTimeDependentTerms(t, *dynamicField_);
    }

  return CuVectorParameter(this);
}
