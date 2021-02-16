#pragma once

#include <curand.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dmitensor.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "strayfield.hpp"
#include "variable.hpp"
#include "world.hpp"

class FieldQuantity;
class MumaxWorld;
class System;

class Ferromagnet {
 public:
  Ferromagnet(MumaxWorld* world,
              Grid grid,
              std::string name,
              GpuBuffer<bool> geometry = GpuBuffer<bool>());
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;  // TODO: check if default is ok

  std::string name() const;
  std::shared_ptr<const System> system() const;
  const World* world() const;
  Grid grid() const;
  real3 cellsize() const;
  const Variable* magnetization() const;
  const GpuBuffer<bool>& getGeometry() const;

  const StrayField* getStrayField(const Ferromagnet*) const;
  std::vector<const StrayField*> getStrayFields() const;
  void addStrayField(
      const Ferromagnet*,
      StrayFieldExecutor::Method method = StrayFieldExecutor::METHOD_AUTO);
  void removeStrayField(const Ferromagnet*);

  void minimize(real tol = 1e-6, int nSamples = 10);

 private:
  Ferromagnet(const Ferromagnet&);
  Ferromagnet& operator=(const Ferromagnet&);

 private:
  std::shared_ptr<System>
      system_;  // the system_ has to be initialized first,
                // hence its listed as the first datamember here
  NormalizedVariable magnetization_;
  std::map<const Ferromagnet*, StrayField*> strayFields_;
  std::string name_;

 public:
  bool enableDemag;
  VectorParameter anisU;
  VectorParameter jcur;
  /** Uniform bias magnetic field which will affect a ferromagnet.
   * Measured in Teslas.
   */
  VectorParameter biasMagneticField;
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

  DmiTensor dmiTensor;
};
