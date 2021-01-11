#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "parameter.hpp"

Parameter::Parameter(std::shared_ptr<const System> system, real value)
    : system_(system), field_(nullptr), uniformValue_(value) {}

Parameter::~Parameter() {
  if (field_)
    delete field_;

  removeAllTimeDependentTerms();
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

void Parameter::addTimeDependentTerm(const time_function& term) {
    time_dep_terms.emplace_back(term, nullptr);
}

void Parameter::addTimeDependentTerm(const time_function& term, const Field& mask) {
    // time_dep_terms.push_back(std::make_pair(term, std::make_unique<Field>(mask)));
    // check whether it creates a copy of time_func here via & == &
    auto mask_ = new Field(mask);
    time_dep_terms.emplace_back(term, mask_);
}

void Parameter::removeAllTimeDependentTerms() {
    for (auto& term : time_dep_terms) {
        auto mask_ = std::get<Field*>(term);
        delete mask_;
    }

    time_dep_terms.clear();
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

Field Parameter::evalTimeDependentTerms(real t) const {
    Field p(system_, ncomp());
    p.setUniformComponent(0, 0);

    for (auto& term : time_dep_terms) {
        auto& func = std::get<Parameter::time_function>(term);
        auto mask = std::get<Field*>(term);

        if (mask) {
            p += func(t) * (*mask);
        }
        else {
            Field f(system_, ncomp());
            f.setUniformComponent(0, func(t));
            p += f;
        }
    }

    return p;
}

Field Parameter::eval() const {
    auto t = system_->world()->time();
  Field p(system_, ncomp());
  Field f = evalTimeDependentTerms(t);

  if (field_) {
      p = *field_;
  }
  else {
      p.setUniformComponent(0, uniformValue_);
  }

  p += f;

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
