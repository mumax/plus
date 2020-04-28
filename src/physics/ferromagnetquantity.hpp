#pragma once

#include "fieldquantity.hpp"
#include "scalarquantity.hpp"

class Ferromagnet;

class FerromagnetFieldQuantity : public FieldQuantity {
 public:
  FerromagnetFieldQuantity(Ferromagnet*,
                      int ncomp,
                      std::string name,
                      std::string unit);
  int ncomp() const;
  Grid grid() const;
  std::string name() const;
  std::string unit() const;

 protected:
  Ferromagnet* ferromagnet_;
  int ncomp_;
  std::string name_;
  std::string unit_;
};

class FerromagnetScalarQuantity : public ScalarQuantity {
 public:
  FerromagnetScalarQuantity(Ferromagnet*,
                      std::string name,
                      std::string unit);
  std::string name() const;
  std::string unit() const;

 protected:
  Ferromagnet* ferromagnet_;
  std::string name_;
  std::string unit_;
};