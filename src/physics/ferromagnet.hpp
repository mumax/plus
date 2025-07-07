#pragma once

#include <curand.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "dmitensor.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "inter_parameter.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "poissonsystem.hpp"
#include "variable.hpp"
#include "world.hpp"
#include "system.hpp"

class HostMagnet;

class Ferromagnet : public Magnet {
 public:
  Ferromagnet(std::shared_ptr<System> system_ptr,
              std::string name,
              HostMagnet* hostMagnet_ = nullptr);

  Ferromagnet(MumaxWorld* world,
              Grid grid,
              std::string name,
              GpuBuffer<bool> geometry,
              GpuBuffer<unsigned int> regions);
  ~Ferromagnet() override;

  const Variable* magnetization() const;

  bool isSublattice() const;

  const HostMagnet* hostMagnet() const { return hostMagnet_; }

  void minimize(real tol = 1e-6, int nSamples = 10);
  void relax(real tol);

 private:
  NormalizedVariable magnetization_;

  HostMagnet* hostMagnet_;

 public:
  mutable PoissonSystem poissonSystem;
  bool enableDemag;
  bool enableOpenBC;
  bool enableZhangLiTorque;
  bool enableSlonczewskiTorque;
  bool fixedLayerOnTop;
  VectorParameter anisU;
  VectorParameter anisC1;
  VectorParameter anisC2;
  VectorParameter jcur;
  VectorParameter fixedLayer;
  /** Uniform bias magnetic field which will affect a ferromagnet.
   * Measured in Teslas.
   */
  VectorParameter biasMagneticField;
  Parameter msat;
  Parameter aex;
  InterParameter interExch;
  InterParameter scaleExch;
  Parameter ku1;
  Parameter ku2;
  Parameter kc1;
  Parameter kc2;
  Parameter kc3;
  Parameter alpha;
  Parameter temperature;
  Parameter Lambda;
  Parameter freeLayerThickness;
  Parameter epsilonPrime;
  Parameter xi;
  Parameter pol;
  Parameter appliedPotential;
  Parameter conductivity;
  Parameter amrRatio;
  Parameter frozenSpins;
  real RelaxTorqueThreshold;
  
  curandGenerator_t randomGenerator;

  DmiTensor dmiTensor;

  // Magnetoelasticity
  Parameter B1;  // First magnetoelastic coupling constant
  Parameter B2;  // Second magnetoelastic coupling constant
};