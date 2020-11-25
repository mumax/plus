#include "ferromagnet.hpp"

#include <curand.h>

#include <random>

#include "fieldquantity.hpp"
#include "magnetfield.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "system.hpp"

Ferromagnet::Ferromagnet(MumaxWorld* world, Grid grid, std::string name)
    : system_(new System(world, grid)),
      magnetization_(name + ":magnetization", "", system_.get(), 3),
      aex(system_.get(), 0.0),
      msat(system_.get(), 1.0),
      ku1(system_.get(), 0.0),
      ku2(system_.get(), 0.0),
      alpha(system_.get(), 0.0),
      temperature(system_.get(), 0.0),
      idmi(system_.get(), 0.0),
      xi(system_.get(), 0.0),
      pol(system_.get(), 0.0),
      anisU(system_.get(), {0, 0, 0}),
      jcur(system_.get(), {0, 0, 0}),
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
  for (auto& entry : magnetFields_) {
    delete entry.second;
  }
  curandDestroyGenerator(randomGenerator);
}

std::string Ferromagnet::name() const {
  return name_;
}

System* Ferromagnet::system() const {
  return system_.get();
}

World* Ferromagnet::world() const {
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

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

const MagnetField* Ferromagnet::getMagnetField(
    const Ferromagnet* magnet) const {
  auto it = magnetFields_.find(magnet);
  if (it == magnetFields_.end())
    return nullptr;
  return it->second;
}

std::vector<const MagnetField*> Ferromagnet::getMagnetFields() const {
  std::vector<const MagnetField*> magnetFields;
  magnetFields.reserve(magnetFields_.size());
  for (const auto& entry : magnetFields_) {
    magnetFields.push_back(entry.second);
  }
  return magnetFields;
}

void Ferromagnet::addMagnetField(const Ferromagnet* magnet,
                                 MagnetFieldComputationMethod method) {
  if (world() != magnet->world()) {
    throw std::runtime_error(
        "Can not define the field of the magnet on this magnet because it is "
        "not in the same world.");
  }

  auto it = magnetFields_.find(magnet);
  if (it != magnetFields_.end()) {
    // MagnetField is already registered, just need to update the method
    it->second->setMethod(method);
    return;
  }

  // Stray field of magnet (parameter) on this magnet (the object)
  magnetFields_[magnet] = new MagnetField(magnet, system(), method);
}

void Ferromagnet::removeMagnetField(const Ferromagnet* magnet) {
  auto it = magnetFields_.find(magnet);
  if (it != magnetFields_.end()) {
    delete it->second;
    magnetFields_.erase(it);
  }
}