#include "ferromagnet.hpp"

#include <curand.h>
#include <chrono>

#include <memory>
#include <random>
#include <math.h>
#include <cfloat>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "poissonsystem.hpp"
#include "relaxer.hpp"

Ferromagnet::Ferromagnet(std::shared_ptr<System> system_ptr, 
                         std::string name,
                         Antiferromagnet* hostMagnet)
    : Magnet(system_ptr, name),
      hostMagnet_(hostMagnet),
      magnetization_(system(), 3, name + ":magnetization", ""),
      msat(system(), 1.0, name + ":msat", "A/m"),
      aex(system(), 0.0, name + ":aex", "J/m"),
      interExch(system(), 0.0, name + ":inter_exchange", "J/m"),
      scaleExch(system(), 1.0, name + ":scale_exchange", ""),
      ku1(system(), 0.0, name + ":ku1", "J/m3"),
      ku2(system(), 0.0, name + "ku2", "J/m3"),
      kc1(system(), 0.0, name + ":kc1", "J/m3"),
      kc2(system(), 0.0, name + ":kc2", "J/m3"),
      kc3(system(), 0.0, name + "kc3", "J/m3"),
      alpha(system(), 0.0, name + ":alpha", ""),
      temperature(system(), 0.0, name + ":temperature", "K"),
      xi(system(), 0.0, name + ":xi", ""),
      Lambda(system(), 0.0, name + ":lambda", ""),
      freeLayerThickness(system(), grid().size().z * cellsize().z,
                         name + ":free_layer_thickness", "m"),
      fixedLayerOnTop(true),
      epsilonPrime(system(), 0.0, name + ":epsilon_prime", ""),
      fixedLayer(system(), {0, 0, 0}, name + ":fixed_layer", ""),
      pol(system(), 0.0, name + ":pol", ""),
      anisU(system(), {0, 0, 0}, name + ":anisU", ""),
      anisC1(system(), {0, 0, 0}, name + ":anisC1", ""),
      anisC2(system(), {0, 0, 0}, name + ":anisC2", ""),
      jcur(system(), {0, 0, 0}, name + ":jcur", "A/m2"),
      biasMagneticField(system(), {0, 0, 0}, name + ":bias_magnetic_field", "T"),
      dmiTensor(system()),
      enableDemag(true),
      enableOpenBC(false),
      enableZhangLiTorque(true),
      enableSlonczewskiTorque(true),
      appliedPotential(system(), std::nanf("0"), name + ":applied_potential", "V"),
      conductivity(system(), 0.0, name + ":conductivity", "S/m"),
      amrRatio(system(), 0.0, name + ":amr_ratio", ""),
      RelaxTorqueThreshold(-1.0),
      poissonSystem(this), 
      // magnetoelasticity
      B1(system(), 0.0, name + ":B1", "J/m3"),
      B2(system(), 0.0, name + ":B1", "J/m3") {
    {// Initialize random magnetization
    // TODO: this can be done much more efficient somewhere else
    int nvalues = 3 * this->grid().ncells();
    std::vector<real> randomValues(nvalues);
    std::normal_distribution<real> dist(0.0, 1.0);
    std::default_random_engine randomEngine;
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    randomEngine.seed (seed);
    for (auto& v : randomValues) {
      v = dist(randomEngine);
    }
    Field randomField(system(), 3);
    randomField.setData(&randomValues[0]);
    magnetization_.set(randomField);
  }
  // Initialize CUDA RNG
  // TODO: move the generator to somewhere else
  curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randomGenerator,
  static_cast<int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
}

Ferromagnet::Ferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry,
                         GpuBuffer<unsigned int> regions)
    : Ferromagnet(std::make_shared<System>(world, grid, geometry, regions), name) {}

Ferromagnet::~Ferromagnet() {
  curandDestroyGenerator(randomGenerator);
}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

bool Ferromagnet::isSublattice() const {
  return !(hostMagnet_ == nullptr);
}

const Antiferromagnet* Ferromagnet::hostMagnet() const {
  return hostMagnet_;
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void Ferromagnet::relax(real tol) {
  real threshold = this->RelaxTorqueThreshold;
  Relaxer relaxer(this, {threshold}, tol);
  relaxer.exec();
}
