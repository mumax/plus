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

  void checkGridCompatibility() const;

  real tipsize;
  real lift;

 private:
  const Grid grid_;
  std::map<std::string, Magnet*> magnets_;
  std::shared_ptr<System> system_;
};
