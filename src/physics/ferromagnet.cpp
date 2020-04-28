#include "ferromagnet.hpp"

#include <random>

#include "fieldquantity.hpp"
#include "minimizer.hpp"

Ferromagnet::Ferromagnet(World* world, std::string name, Grid grid)
    : System(world, name, grid),
      demagField_(this),
      anisotropyField_(this),
      anisotropyEnergyDensity_(this),
      exchangeField_(this),
      exchangeEnergyDensity_(this),
      effectiveField_(this),
      torque_(this),
      magnetization_(name + ":magnetization", "", 3, grid),
      aex(grid, 0.0),
      msat(grid, 1.0),
      ku1(grid, 0.0),
      alpha(grid, 0.0),
      anisU(grid, {0, 0, 0}) {
  enableDemag = true;
  {
    // TODO: this can be done much more efficient somewhere else
    int ncomp = 3;
    int nvalues = ncomp * grid_.ncells();
    std::vector<real> randomValues(nvalues);
    std::uniform_real_distribution<real> unif(-1, 1);
    std::default_random_engine randomEngine;
    for (auto& v : randomValues) {
      v = unif(randomEngine);
    }
    Field randomField(grid_, ncomp);
    randomField.setData(&randomValues[0]);
    magnetization_.set(&randomField);
  }
}

Ferromagnet::~Ferromagnet() {}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

const FieldQuantity* Ferromagnet::demagField() const {
  return &demagField_;
}

const FieldQuantity* Ferromagnet::anisotropyField() const {
  return &anisotropyField_;
}

const FieldQuantity* Ferromagnet::anisotropyEnergyDensity() const {
  return &anisotropyEnergyDensity_;
}

const FieldQuantity* Ferromagnet::exchangeField() const {
  return &exchangeField_;
}

const FieldQuantity* Ferromagnet::exchangeEnergyDensity() const {
  return &exchangeEnergyDensity_;
}

const FieldQuantity* Ferromagnet::effectiveField() const {
  return &effectiveField_;
}

const FieldQuantity* Ferromagnet::torque() const {
  return &torque_;
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}