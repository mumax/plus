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
#include "poissonsystem.hpp"
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
              int ncomp,
              std::string name,
              GpuBuffer<bool> geometry = GpuBuffer<bool>());
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;  // TODO: check if default is ok

  std::string name() const;
  std::shared_ptr<const System> system() const;
  int ncomp() const;
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
  void relax();

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
  int ncomp_;

 public:
  mutable PoissonSystem poissonSystem;

  bool enableDemag;
  bool enableOpenBC;
  FM_VectorParameter anisU;
  FM_VectorParameter anisC1;
  FM_VectorParameter anisC2;
  FM_VectorParameter jcur;
  FM_VectorParameter FixedLayer;
  /** Uniform bias magnetic field which will affect a ferromagnet.
   * Measured in Teslas.
   */
  FM_VectorParameter biasMagneticField;
  Parameter msat;
  Parameter msat2;
  Parameter aex;
  Parameter aex2;
  Parameter afmex_cell;
  Parameter afmex_nn;
  Parameter ku1;
  Parameter ku12;
  Parameter ku2;
  Parameter ku22;
  Parameter kc1;
  Parameter kc2;
  Parameter kc3;
  Parameter kc12;
  Parameter kc22;
  Parameter kc32;
  Parameter alpha;
  Parameter temperature;
  Parameter idmi;
  Parameter latcon;
  Parameter Lambda;
  Parameter FreeLayerThickness;
  Parameter eps_prime;
  Parameter xi;
  Parameter pol;
  Parameter appliedPotential;
  Parameter conductivity;
  Parameter conductivity2;
  Parameter amrRatio;
  Parameter amrRatio2;
  real RelaxTorqueThreshold;

  curandGenerator_t randomGenerator;

  DmiTensor dmiTensor;
};
