#pragma once

#include <curand.h>

#include <map>
#include <string>
#include <vector>

#include "field.hpp"
#include "grid.hpp"
#include "handler.hpp"
#include "magnetfield.hpp"
#include "parameter.hpp"
#include "ref.hpp"
#include "system.hpp"
#include "variable.hpp"

class FieldQuantity;
class MumaxWorld;

class Ferromagnet : public System {
 public:
  Ferromagnet(std::string name, Grid grid);
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;  // TODO: check if default is ok

  const Variable* magnetization() const;

  bool enableDemag;

  VectorParameter anisU;
  VectorParameter jcur;
  Parameter msat;
  Parameter aex;
  Parameter ku1;
  Parameter ku2;
  Parameter alpha;
  Parameter temperature;
  Parameter idmi;
  Parameter xi;
  Parameter pol;

  const MagnetField* getMagnetField(const Ferromagnet*) const;
  std::vector<const MagnetField*> getMagnetFields() const;
  void addMagnetField(
      const Ferromagnet*,
      MagnetFieldComputationMethod method = MAGNETFIELDMETHOD_BRUTE);
  void removeMagnetField(const Ferromagnet*);

  void minimize(real tol = 1e-6, int nSamples = 10);

 private:
  Ferromagnet(const Ferromagnet&);
  Ferromagnet& operator=(const Ferromagnet&);

 private:
  NormalizedVariable magnetization_;
  std::map<const Ferromagnet*, MagnetField*> magnetFields_;

 public:
  curandGenerator_t randomGenerator;
};