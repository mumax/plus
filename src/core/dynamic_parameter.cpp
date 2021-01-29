#include "dynamic_parameter.hpp"

#include "fieldops.hpp"

template <typename T>
DynamicParameter<T>::DynamicParameter(const DynamicParameter<T>& other)
    : dynamicField_(nullptr) {
  if (other.dynamicField_) {
    dynamicField_.reset(new Field(*other.dynamicField_));
  }
  if (!other.time_dep_terms.empty()) {
    time_dep_terms = other.time_dep_terms;
  }
}

template <typename T>
DynamicParameter<T>& DynamicParameter<T>::operator=(
    const DynamicParameter<T>& other) {
  if (other.dynamicField_) {
    dynamicField_.reset(new Field(*other.dynamicField_));
  }
  if (!other.time_dep_terms.empty()) {
    time_dep_terms = other.time_dep_terms;
  }

  return *this;
}

template <typename T>
DynamicParameter<T>::DynamicParameter(
    const DynamicParameter<T>&& other) noexcept
    : dynamicField_(std::move(other.dynamicField_)),
      time_dep_terms(std::move(other.time_dep_terms)) {}

template <typename T>
DynamicParameter<T>& DynamicParameter<T>::operator=(
    const DynamicParameter<T>&& other) noexcept {
  dynamicField_ = std::move(other.dynamicField_);
  time_dep_terms = std::move(other.time_dep_terms);

  return *this;
}

template <typename T>
void DynamicParameter<T>::evalTimeDependentTerms(real t, Field& p) const {
  for (auto& term : time_dep_terms) {
    auto& func = std::get<std::function<T(real)>>(term);
    auto& mask = std::get<Field>(term);

    if (!mask.empty()) {
      p += func(t) * mask;
    } else {
      Field f(p.system(), p.ncomp());
      f.setUniformValue(func(t));
      p += f;
    }
  }
}

template <typename T>
bool DynamicParameter<T>::isUniform() const {
  for (auto& term : time_dep_terms) {
    auto& mask = std::get<Field>(term);

    if (!mask.empty()) {
      return false;
    }
  }

  return true;
}

template class DynamicParameter<real>;
template class DynamicParameter<real3>;
