#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;

class MFM : public FieldQuantity {
 public:
  MFM(const Magnet*, Grid grid);

  Field eval() const;

  int ncomp() const;

  std::shared_ptr<const System> system() const;

  void setLift(real value);

  real tipsize;

 private:
  real lift_;
  const Grid grid_;
  const Magnet* magnet_;
  std::shared_ptr<System> system_;
};
