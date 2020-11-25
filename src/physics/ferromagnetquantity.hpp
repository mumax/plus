#pragma once

#include <functional>

#include "ferromagnet.hpp"
#include "fieldquantity.hpp"
#include "handler.hpp"
#include "ref.hpp"
#include "scalarquantity.hpp"
#include "system.hpp"

class Ferromagnet;

typedef std::function<Field(const Ferromagnet*)> FM_FieldFunc;
typedef std::function<real(const Ferromagnet*)> FM_ScalarFunc;

class FM_FieldQuantity : public FieldQuantity {
 public:
  FM_FieldQuantity(const Ferromagnet* ferromagnet,
                   FM_FieldFunc evalfunc,
                   int ncomp,
                   std::string name,
                   std::string unit)
      : ferromagnet_(ferromagnet),
        evalfunc_(evalfunc),
        ncomp_(ncomp),
        name_(name),
        unit_(unit) {}

  FM_FieldQuantity* clone() {
    return new FM_FieldQuantity(ferromagnet_, evalfunc_, ncomp_, name_, unit_);
  }

  int ncomp() const { return ncomp_; }
  const System* system() const { return ferromagnet_->system(); }
  std::string name() const { return name_; }
  std::string unit() const { return unit_; }

  Field eval() const { return evalfunc_(ferromagnet_); }
  Field operator()() const { return evalfunc_(ferromagnet_); }

 private:
  const Ferromagnet* ferromagnet_;
  int ncomp_;
  std::string name_;
  std::string unit_;
  FM_FieldFunc evalfunc_;
};

class FM_ScalarQuantity : public ScalarQuantity {
 public:
  FM_ScalarQuantity(const Ferromagnet* ferromagnet,
                    FM_ScalarFunc evalfunc,
                    std::string name,
                    std::string unit)
      : ferromagnet_(ferromagnet),
        evalfunc_(evalfunc),
        name_(name),
        unit_(unit) {}

  std::string name() const { return name_; }
  std::string unit() const { return unit_; }
  real eval() const { return evalfunc_(ferromagnet_); }

 private:
  const Ferromagnet* ferromagnet_;
  std::string name_;
  std::string unit_;
  FM_ScalarFunc evalfunc_;
};