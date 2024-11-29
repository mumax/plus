#include "magnet.hpp"

#include <curand.h>

#include <memory>
#include <random>
#include <math.h>
#include <cfloat>
#include <stdexcept>

#include "antiferromagnet.hpp"
#include "ferromagnet.hpp"
#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "mumaxworld.hpp"
#include "relaxer.hpp"
#include "strayfield.hpp"

Magnet::Magnet(std::shared_ptr<System> system_ptr,
               std::string name)
    : system_(system_ptr),
      name_(name),
      enableAsStrayFieldSource(true),
      enableAsStrayFieldDestination(true),
      // elasticity
      enableElastodynamics_(false),
      externalBodyForce(system(), {0, 0, 0}, name + ":external_body_force", "N/m3"),
      c11(system(), 0.0, name + ":c11", "N/m2"),
      c12(system(), 0.0, name + ":c12", "N/m2"),
      c44(system(), 0.0, name + ":c44", "N/m2"),
      eta(system(), 0.0, name + ":eta", "kg/m3s"),
      rho(system(), 1.0, name + ":rho", "kg/m3") {
  // Check that the system has at least size 1
  int3 size = system_->grid().size();
  if (size.x < 1 || size.y < 1 || size.z < 1)
    throw std::invalid_argument("The grid of a magnet should have size >= 1 "
                                "in all directions.");
}

Magnet::~Magnet() {
  // TODO: stray field pointers should be smart
  for (auto& entry : strayFields_) {
    delete entry.second;
  }
}

// TODO: add copies for all variables?
Magnet::Magnet(Magnet&& other) noexcept
    : system_(other.system_),
      name_(other.name_),
      
      externalBodyForce(other.externalBodyForce),
      c11(other.c11), c12(other.c12), c44(other.c44),
      eta(other.eta), rho(other.rho) {
  other.system_ = nullptr;
  other.name_ = "";
}

// Provide move constructor and move assignment operator
// TODO: add copies for all variables?
Magnet& Magnet::operator=(Magnet&& other) noexcept {
      if (this != &other) {
          system_ = other.system_;
          name_ = other.name_;
          other.system_ = nullptr;
          other.name_ = "";

        // TODO: add reset to `other` of some kind? idk
        externalBodyForce = other.externalBodyForce;
        c11 = other.c11;
        c12 = other.c12;
        c44 = other.c44;
        eta = other.eta;
        rho = other.rho;
      }
      return *this;
  }

std::string Magnet::name() const {
  return name_;
}

std::shared_ptr<const System> Magnet::system() const {
  return system_;
}

const World* Magnet::world() const {
  return system()->world();
}

const MumaxWorld* Magnet::mumaxWorld() const {
  // static_cast: no check needed, world() is always a MumaxWorld
  return static_cast<const MumaxWorld*>(this->world());
}

Grid Magnet::grid() const {
  return system()->grid();
}

real3 Magnet::cellsize() const {
  return world()->cellsize();
}

const GpuBuffer<bool>& Magnet::getGeometry() const {
  return system_->geometry();
}

const Ferromagnet* Magnet::asFM() const {
  return dynamic_cast<const Ferromagnet*>(this);
}

const Antiferromagnet* Magnet::asAFM() const {
  return dynamic_cast<const Antiferromagnet*>(this);
}

const StrayField* Magnet::getStrayField(const Magnet* magnet) const {
  auto it = strayFields_.find(magnet);
  if (it == strayFields_.end())
    return nullptr;
  return it->second;
}

std::vector<const StrayField*> Magnet::getStrayFields() const {
  std::vector<const StrayField*> strayFields;
  strayFields.reserve(strayFields_.size());
  for (const auto& entry : strayFields_) {
    strayFields.push_back(entry.second);
  }
  return strayFields;
}

void Magnet::addStrayField(const Magnet* magnet,
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

void Magnet::removeStrayField(const Magnet* magnet) {
  auto it = strayFields_.find(magnet);
  if (it != strayFields_.end()) {
    delete it->second;
    strayFields_.erase(it);
  }
}

const Variable* Magnet::elasticDisplacement() const {
  if (!this->enableElastodynamics()) {
    throw std::domain_error("elasticDisplacement Variable does not exist yet. "
                            "Enable elastodynamics first.");
  }
  return elasticDisplacement_.get();
}

const Variable* Magnet::elasticVelocity() const {
  if (!this->enableElastodynamics()) {
    throw std::domain_error("elasticVelocity Variable does not exist yet. "
                            "Enable elastodynamics first.");
  }
  return elasticVelocity_.get();
}

void Magnet::setEnableElastodynamics(bool value) {
  // if this is a ferromagnetic sublattice, stop!
  auto thisFM = this->asFM();
  if (thisFM) {
    if (thisFM->isSublattice()) {
      throw std::invalid_argument(
        "Cannot enable/disable elastodynamics for a sublattice.");
    }
  }

  if (enableElastodynamics_ != value) {
    enableElastodynamics_ = value;

    if (value) {
      // properly initialize Variables now
      elasticDisplacement_ = std::make_unique<Variable>(system(), 3,
                                        name() + ":elastic_displacement", "m");
      elasticDisplacement_->set(real3{0,0,0});
      elasticVelocity_ = std::make_unique<Variable>(system(), 3,
                                        name() + ":elastic_velocity", "m/s");
      elasticVelocity_->set(real3{0,0,0});
    } else {
      // free memory of unnecessary Variables
      elasticDisplacement_.reset();
      elasticVelocity_.reset();
    }

    this->mumaxWorld()->resetTimeSolverEquations();
  }
}
