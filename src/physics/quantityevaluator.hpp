#pragma once

#include <functional>
#include <memory>
#include <string>

#include "antiferromagnet.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "magnet.hpp"
#include "ncafm.hpp"
#include "scalarquantity.hpp"
#include "system.hpp"


// ========== FieldQuantityEvaluator ==========

template <class T>  // meant to be Magnet, Ferromagnet, Antiferromagnet or NcAfm
class FieldQuantityEvaluator : public FieldQuantity {
 public:
  FieldQuantityEvaluator(const T* ptr,
                         std::function<Field(const T*)> evalfunc,
                         int ncomp,
                         std::string name,
                         std::string unit,
                         std::function<bool(const T*)> assuredZeroFunc =
                           [](const T* ptr){ return false; })
      : ptr_(ptr),
        evalfunc_(evalfunc),
        ncomp_(ncomp),
        name_(name),
        unit_(unit),
        assuredZeroFunc_(assuredZeroFunc) {}

  FieldQuantityEvaluator<T>* clone() {
    return new FieldQuantityEvaluator<T>(ptr_, evalfunc_, ncomp_, name_, unit_,
                                         assuredZeroFunc_);
  }

  int ncomp() const { return ncomp_; }
  std::shared_ptr<const System> system() const { return ptr_->system(); }
  std::string name() const { return name_; }
  std::string unit() const { return unit_; }

  Field eval() const {
    if (assuredZero()) { return Field(system(), ncomp(), 0); }
    return evalfunc_(ptr_);
  }
  Field operator()() const { return this->eval(); }

  bool assuredZero() const { return assuredZeroFunc_(ptr_); }

 private:
  const T* ptr_;  // meant to be Magnet, Ferromagnet, Antiferromagnet or NcAfm
  int ncomp_;
  std::string name_;
  std::string unit_;
  std::function<Field(const T*)> evalfunc_;
  std::function<bool(const T*)> assuredZeroFunc_;
};

using AFM_FieldQuantity = FieldQuantityEvaluator<Antiferromagnet>;
using FM_FieldQuantity = FieldQuantityEvaluator<Ferromagnet>;
using M_FieldQuantity = FieldQuantityEvaluator<Magnet>;
using NcAfm_FieldQuantity = FieldQuantityEvaluator<NcAfm>;


// ========== ScalarQuantityEvaluator ==========

template <class T>
class ScalarQuantityEvaluator : public ScalarQuantity {
 public:
  ScalarQuantityEvaluator(const T* ptr,
                          std::function<real(const T*)> evalfunc,
                          std::string name,
                          std::string unit,
                          std::function<bool(const T*)> assuredZeroFunc =
                            [](const T* ptr){ return false; })
      : ptr_(ptr),
        evalfunc_(evalfunc),
        name_(name),
        unit_(unit),
        assuredZeroFunc_(assuredZeroFunc) {}

  std::string name() const { return name_; }
  std::string unit() const { return unit_; }
  
  real eval() const {
    if (assuredZero()) { return 0; }
    return evalfunc_(ptr_);
  }

  bool assuredZero() const { return assuredZeroFunc_(ptr_); }

 private:
  const T* ptr_;
  std::string name_;
  std::string unit_;
  std::function<real(const T*)> evalfunc_;
  std::function<bool(const T*)> assuredZeroFunc_;
};

using AFM_ScalarQuantity = ScalarQuantityEvaluator<Antiferromagnet>;
using FM_ScalarQuantity = ScalarQuantityEvaluator<Ferromagnet>;
using M_ScalarQuantity = ScalarQuantityEvaluator<Magnet>;
using NcAfm_ScalarQuantity = ScalarQuantityEvaluator<NcAfm>;