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
#include "variable.hpp"
#include "world.hpp"

class FieldQuantity;
class MumaxWorld;
class System;

class Ferromagnet {
 public:
  Ferromagnet(MumaxWorld* world, Grid grid, std::string name);
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;  // TODO: check if default is ok

  std::string name() const;
  std::shared_ptr<System> system() const;
  World* world() const;
  Grid grid() const;
  real3 cellsize() const;
  const Variable* magnetization() const;

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
  std::shared_ptr<System>
      system_;  // the system_ has to be initialized first,
                // hence its listed as the first datamember here
  NormalizedVariable magnetization_;
  std::map<const Ferromagnet*, MagnetField*> magnetFields_;
  std::string name_;

 public:
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
  curandGenerator_t randomGenerator;
};