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
  return !staticField_ && DynamicParameter<real>::isUniform();
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
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);

    staticField += dynamicField;
  }

  return staticField;
}

real Parameter::getUniformValue() const {
  if (!isUniform()) {
    throw std::invalid_argument("Cannot get uniform value of non-uniform Parameter.");
  }
  return uniformValue_;
}

CuParameter Parameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
  }

  return CuParameter(this);
}

FM_VectorParameter::FM_VectorParameter(std::shared_ptr<const System> system,
                                 real3 value)
    : system_(system), staticField_(nullptr), uniformValue_(value) {}

AFM_VectorParameter::AFM_VectorParameter(std::shared_ptr<const System> system,
                                 real6 value)
    : system_(system), staticField_(nullptr), uniformValue_(value) {}

FM_VectorParameter::~FM_VectorParameter() {
  if (staticField_)
    delete staticField_;
}

AFM_VectorParameter::~AFM_VectorParameter() {
  if (staticField_)
    delete staticField_;
}

void FM_VectorParameter::set(real3 value) {
  uniformValue_ = value;
  if (staticField_)
    delete staticField_;
}

void AFM_VectorParameter::set(real6 value) {
  uniformValue_ = value;
  if (staticField_)
    delete staticField_;
}

void FM_VectorParameter::set(const Field& values) {
  staticField_ = new Field(values);
}

void AFM_VectorParameter::set(const Field& values) {
  staticField_ = new Field(values);
}

bool FM_VectorParameter::isUniform() const {
  return !staticField_ && DynamicParameter<real3>::isUniform();
}

bool AFM_VectorParameter::isUniform() const {
  return !staticField_ && DynamicParameter<real6>::isUniform();
}

bool FM_VectorParameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == real3{0.0, 0.0, 0.0};
}

bool AFM_VectorParameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == real6{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

int FM_VectorParameter::ncomp() const {
  return 3;
}

int AFM_VectorParameter::ncomp() const {
  return 6;
}

std::shared_ptr<const System> FM_VectorParameter::system() const {
  return system_;
}

std::shared_ptr<const System> AFM_VectorParameter::system() const {
  return system_;
}

Field FM_VectorParameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);

    staticField += dynamicField;
  }

  return staticField;
}

Field AFM_VectorParameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);

    staticField += dynamicField;
  }

  return staticField;
}

FM_CuVectorParameter FM_VectorParameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
  }

  return FM_CuVectorParameter(this);
}

AFM_CuVectorParameter AFM_VectorParameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
  }

  return AFM_CuVectorParameter(this);
}