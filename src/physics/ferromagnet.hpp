#pragma once

#include <map>
#include <string>

#include "anisotropy.hpp"
#include "effectivefield.hpp"
#include "exchange.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "demag.hpp"
#include "quantity.hpp"
#include "system.hpp"
#include "torque.hpp"
#include "variable.hpp"

class World;

class Ferromagnet : public System {
 public:
  Ferromagnet(World* world, std::string name, Grid grid);
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;

  const Variable* magnetization() const;

  real3 anisU;
  real msat, ku1, aex, alpha;

  const Quantity* demagField() const;
  const Quantity* anisotropyField() const;
  const Quantity* exchangeField() const;
  const Quantity* effectiveField() const;
  const Quantity* torque() const;

 private:
  Ferromagnet(const Ferromagnet&);
  Ferromagnet& operator=(const Ferromagnet&);

 private:
  NormalizedVariable magnetization_;

  DemagField demagField_;
  AnisotropyField anisotropyField_;
  ExchangeField exchangeField_;
  EffectiveField effectiveField_;
  Torque torque_;
};