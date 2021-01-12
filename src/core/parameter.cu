#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "parameter.hpp"

Parameter::Parameter(std::shared_ptr<const System> system, real value)
    : system_(system), staticField_(nullptr), dynamicField_(nullptr), uniformValue_(value) {}

Parameter::~Parameter() {
    if (staticField_)
        delete staticField_;
    if (dynamicField_)
        delete dynamicField_;
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

void Parameter::addTimeDependentTerm(const time_function& term) {
    time_dep_terms.emplace_back(term, Field());
}

void Parameter::addTimeDependentTerm(const time_function& term, const Field& mask) {
    // time_dep_terms.push_back(std::make_pair(term, std::make_unique<Field>(mask)));
    // check whether it creates a copy of time_func here via & == &
    time_dep_terms.emplace_back(time_function(term), Field(mask));
    // time_dep_terms.push_back(std::make_pair(time_function(term), Field(mask)));
}

void Parameter::removeAllTimeDependentTerms() {
    time_dep_terms.clear();
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

void Parameter::evalTimeDependentTerms(real t, Field& p) const {
    for (auto& term : time_dep_terms) {
        auto& func = std::get<Parameter::time_function>(term);
        auto& mask = std::get<Field>(term);

        if (!mask.empty()) {
            p += func(t) * mask;
        }
        else {
            Field f(system_, ncomp());
            f.setUniformComponent(0, func(t));
            p += f;
        }
    }
}

Field Parameter::eval() const {
  auto t = system_->world()->time();
  Field static_field(system_, ncomp());
  Field dynamic_field(system_, ncomp());

  dynamic_field.setUniformComponent(0, 0);
  evalTimeDependentTerms(t, dynamic_field);

  if (staticField_) {
      static_field = *staticField_;
  }
  else {
      static_field.setUniformComponent(0, uniformValue_);
  }

  static_field += dynamic_field;

  return static_field;
}

CuParameter Parameter::cu() const {
    auto t = system_->world()->time();

    dynamicField_ = new Field(system_, ncomp());
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
