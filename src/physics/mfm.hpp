#pragma once

#include "quantityevaluator.hpp"
#include "mumaxworld.hpp"
#include <map>

class MFM : public FieldQuantity {
 public:
  MFM(Magnet*, Grid grid);
  MFM(const MumaxWorld*, Grid grid);

  Field eval() const;

  int ncomp() const;

  std::shared_ptr<const System> system() const;

  void setLift(real value);

  real tipsize;

 private:
  real lift_;
  const Grid grid_;
  std::map<std::string, Magnet*> magnets_;
  std::shared_ptr<System> system_;
};
