#pragma once

#include <functional>
#include <optional>
#include <memory>
#include <string>

#include "ferromagnet.hpp"
#include "fieldquantity.hpp"
#include "scalarquantity.hpp"
#include "system.hpp"

class Ferromagnet;

typedef std::function<Field(const Ferromagnet*, const int lattice)> FM_FieldFunc;

typedef std::function<real(const Ferromagnet*, const int lattice)> FM_ScalarFunc;

class FM_FieldQuantity : public FieldQuantity {
 public:
  FM_FieldQuantity(const Ferromagnet* ferromagnet,
                   FM_FieldFunc evalfunc,
                   int ncomp,
                   std::string name,
                   std::string unit,
                   int lattice = 0)
      : ferromagnet_(ferromagnet),
        evalfunc_(evalfunc),
        ncomp_(ncomp),
        name_(name),
        unit_(unit) {lattice_ = lattice;}

  FM_FieldQuantity* clone() {
    return new FM_FieldQuantity(ferromagnet_, evalfunc_, ncomp_, name_, unit_, lattice_);
  }

  int ncomp() const { return ncomp_; }
  std::shared_ptr<const System> system() const {
    return ferromagnet_->system();
  }
  std::string name() const { return name_; }
  std::string unit() const { return unit_; }

  Field eval() const { return evalfunc_(ferromagnet_, lattice_); }
  Field operator()() const { return evalfunc_(ferromagnet_, lattice_); }

 private:
  const Ferromagnet* ferromagnet_;
  int ncomp_;
  int lattice_;
  std::string name_;
  std::string unit_;
  FM_FieldFunc evalfunc_;
};

class FM_ScalarQuantity : public ScalarQuantity {
 public:
  FM_ScalarQuantity(const Ferromagnet* ferromagnet,
                    FM_ScalarFunc evalfunc,
                    int lattice,
                    std::string name,
                    std::string unit)
      : ferromagnet_(ferromagnet),
        evalfunc_(evalfunc),
        lattice_(lattice),
        name_(name),
        unit_(unit) {}

  int lattice() const { return lattice_; }
  std::string name() const { return name_; }
  std::string unit() const { return unit_; }
  real eval() const { return evalfunc_(ferromagnet_, lattice_); }
  std::string getSuffixByLattice() const { return (lattice_ == 0) ? "" : ((lattice_ == 1) ? "_sub1" : "_sub2"); }
 private:
  const Ferromagnet* ferromagnet_;
  int lattice_;
  std::string name_;
  std::string unit_;
  FM_ScalarFunc evalfunc_;
};