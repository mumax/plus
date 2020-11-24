#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "system.hpp"

Parameter::Parameter(System* system, real value)
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

bool Parameter::isUniform() const {
  return !field_;
};

bool Parameter::assuredZero() const {
  return isUniform() && uniformValue_ == 0.0;
}

int Parameter::ncomp() const {
  return 1;
}

System* Parameter::system() const {
  return system_;
}

Grid Parameter::grid() const {
  return system_->grid();
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
  if (isUniform())
    return CuParameter(grid(), uniformValue_);
  return CuParameter(grid(), field_->devptr(0));
}

VectorParameter::VectorParameter(System* system, real3 value)
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
};

bool VectorParameter::assuredZero() const {
  return isUniform() && uniformValue_ == real3{0.0, 0.0, 0.0};
}

int VectorParameter::ncomp() const {
  return 3;
}

System* VectorParameter::system() const {
  return system_;
}

Grid VectorParameter::grid() const {
  return system()->grid();
}

Field VectorParameter::eval() const {
  Field p(grid(), ncomp());
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
  if (isUniform())
    return CuVectorParameter(grid(), uniformValue_);
  return CuVectorParameter{grid(), field_->devptr(0), field_->devptr(1),
                           field_->devptr(2)};
}