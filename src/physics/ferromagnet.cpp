#include "ferromagnet.hpp"

#include <curand.h>

#include <random>

#include "fieldquantity.hpp"
#include "magnetfield.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "ref.hpp"

Ferromagnet::Ferromagnet(MumaxWorld* world, Grid grid)
    : System(world, grid),
      magnetization_("magnetization",
                     "",
                     3,
                     grid),  // TODO: add system name to variable name
      aex(this, 0.0),
      msat(this, 1.0),
      ku1(this, 0.0),
      ku2(this, 0.0),
      alpha(this, 0.0),
      temperature(this, 0.0),
      idmi(this, 0.0),
      xi(this, 0.0),
      pol(this, 0.0),
      anisU(this, {0, 0, 0}),
      jcur(this, {0, 0, 0}),
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
    Field randomField(this->grid(), ncomp);
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

  magnetFields_[magnet] = new MagnetField(magnet, grid(), method);
}

void Ferromagnet::removeMagnetField(const Ferromagnet* magnet) {
  auto it = magnetFields_.find(magnet);
  if (it != magnetFields_.end()) {
    delete it->second;
    magnetFields_.erase(it);
  }
}