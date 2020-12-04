#include "ferromagnet.hpp"

#include <curand.h>

#include <memory>
#include <random>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "strayfield.hpp"
#include "system.hpp"

Ferromagnet::Ferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry)
    : system_(new System(world, grid, geometry)),
      magnetization_(name + ":magnetization", "", system_, 3),
      aex(system_, 0.0),
      msat(system_, 1.0),
      ku1(system_, 0.0),
      ku2(system_, 0.0),
      alpha(system_, 0.0),
      temperature(system_, 0.0),
      idmi(system_, 0.0),
      xi(system_, 0.0),
      pol(system_, 0.0),
      anisU(system_, {0, 0, 0}),
      jcur(system_, {0, 0, 0}),
      biasMagneticField(system_, {0, 0, 0}),
      enableDemag(true) {
  {
    // TODO: this can be done much more efficient somewhere else
    int ncomp = 3;
    int nvalues = ncomp * this->grid().ncells();
    std::vector<real> randomValues(nvalues);
    std::uniform_real_distribution<real> unif(-1, 1);
    std::default_random_engine randomEngine;
    for (auto& v : randomValues) {
      v = unif(randomEngine);
    }
    Field randomField(system(), ncomp);
    randomField.setData(&randomValues[0]);
    magnetization_.set(randomField);
  }
  // TODO: move the generator to somewhere else
  curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randomGenerator, 1234);
}

Ferromagnet::~Ferromagnet() {
  for (auto& entry : strayFields_) {
    delete entry.second;
  }
  curandDestroyGenerator(randomGenerator);
}

std::string Ferromagnet::name() const {
  return name_;
}

std::shared_ptr<const System> Ferromagnet::system() const {
  return system_;
}

const World* Ferromagnet::world() const {
  return system()->world();
}

Grid Ferromagnet::grid() const {
  return system()->grid();
}

real3 Ferromagnet::cellsize() const {
  return world()->cellsize();
}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

const GpuBuffer<bool>& Ferromagnet::getGeometry() const {
  return system_->geometry();
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

const StrayField* Ferromagnet::getStrayField(const Ferromagnet* magnet) const {
  auto it = strayFields_.find(magnet);
  if (it == strayFields_.end())
    return nullptr;
  return it->second;
}

std::vector<const StrayField*> Ferromagnet::getStrayFields() const {
  std::vector<const StrayField*> strayFields;
  strayFields.reserve(strayFields_.size());
  for (const auto& entry : strayFields_) {
    strayFields.push_back(entry.second);
  }
  return strayFields;
}

void Ferromagnet::addStrayField(const Ferromagnet* magnet,
                                StrayFieldExecutor::Method method) {
  if (world() != magnet->world()) {
    throw std::runtime_error(
        "Can not define the field of the magnet on this magnet because it is "
        "not in the same world.");
  }

  auto it = strayFields_.find(magnet);
  if (it != strayFields_.end()) {
    // StrayField is already registered, just need to update the method
    it->second->setMethod(method);
    return;
  }

  // Stray field of magnet (parameter) on this magnet (the object)
  strayFields_[magnet] = new StrayField(magnet, system(), method);
}

void Ferromagnet::removeStrayField(const Ferromagnet* magnet) {
  auto it = strayFields_.find(magnet);
  if (it != strayFields_.end()) {
    delete it->second;
    strayFields_.erase(it);
  }
}
