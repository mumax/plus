#pragma once

#include "quantityevaluator.hpp"
#include "mumaxworld.hpp"
#include <map>

class MFM : public FieldQuantity {
 public:
  MFM(Magnet*, Grid grid, std::string name = "");
  MFM(const MumaxWorld*, Grid grid, std::string name = "");

  Field eval() const;

  int ncomp() const;
  std::string name() const {return name_;};
  std::string unit() const {return "J";};

  std::shared_ptr<const System> system() const;

  real tipsize;
  real lift;

 private:
  std::map<std::string, Magnet*> magnets_;
  std::shared_ptr<System> system_;
  std::string name_;
};
