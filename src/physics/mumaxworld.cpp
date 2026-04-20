#include "mumaxworld.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "altermagnet.hpp"
#include "antiferromagnet.hpp"
#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "elastodynamics.hpp"
#include "ferromagnet.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "magnet.hpp"
#include "minimizer.hpp"
#include "ncafm.hpp"
#include "relaxer.hpp"
#include "system.hpp"
#include "thermalnoise.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

MumaxWorld::MumaxWorld(real3 cellsize)
    : World(cellsize),
      biasMagneticField({0, 0, 0}),
      RelaxTorqueThreshold(-1.0) {}

MumaxWorld::MumaxWorld(real3 cellsize, Grid mastergrid, int3 pbcRepetitions)
    : World(cellsize, mastergrid, pbcRepetitions),
      biasMagneticField({0, 0, 0}),
      RelaxTorqueThreshold(-1.0) {}

MumaxWorld::~MumaxWorld() {}

void MumaxWorld::checkAddibility(Grid grid, std::string name) const {
  if (!inMastergrid(grid)) {
      throw std::out_of_range(
          "Can not add magnet because the grid does not fit in the "
          "mastergrid ");
  }

  for (const auto& namedMagnet : magnets_) {
    Magnet* m = namedMagnet.second;
    if (grid.overlaps(m->grid())) {
      throw std::out_of_range(
          "Can not add magnet because it overlaps with another "
          "magnet.");
    }
  }

  if (magnets_.find(name) != magnets_.end()) {
    throw std::runtime_error("A magnet with the name '" + name +
                             "' already exists");
  }
}

Ferromagnet* MumaxWorld::addFerromagnet(Grid grid,
                                        GpuBuffer<bool> geometry,
                                        GpuBuffer<unsigned int> regions,
                                        std::string name) {
  auto mag = addMagnetTempl<Ferromagnet>(ferromagnets_, grid, geometry, regions, name, "ferromagnet");
  resetTimeSolverEquations();
  return mag;
}

Antiferromagnet* MumaxWorld::addAntiferromagnet(Grid grid,
                                                GpuBuffer<bool> geometry,
                                                GpuBuffer<unsigned int> regions,
                                                std::string name) {
  auto mag = addMagnetTempl<Antiferromagnet>(antiferromagnets_, grid, geometry, regions, name, "antiferromagnet");
  hostmagnets_[mag->name()] = mag;
  resetTimeSolverEquations();
  return mag;
}

Altermagnet* MumaxWorld::addAltermagnet(Grid grid,
                                        GpuBuffer<bool> geometry,
                                        GpuBuffer<unsigned int> regions,
                                        std::string name) {
  auto mag = addMagnetTempl<Altermagnet>(altermagnets_, grid, geometry, regions, name, "altermagnet");
  hostmagnets_[mag->name()] = mag;
  resetTimeSolverEquations();
  return mag;
}

NcAfm* MumaxWorld::addNcAfm(Grid grid,
                            GpuBuffer<bool> geometry,
                            GpuBuffer<unsigned int> regions,
                            std::string name) {
  auto mag = addMagnetTempl<NcAfm>(ncafms_, grid, geometry, regions, name, "ncafm");
  hostmagnets_[mag->name()] = mag;
  resetTimeSolverEquations();
  return mag;
}

void MumaxWorld::handleNewStrayfield(Magnet* newMagnet) {
  for (const auto& namedMagnet : magnets_) {
    Magnet* otherMagnet = namedMagnet.second;
    if (otherMagnet != nullptr) {
      otherMagnet->addStrayField(newMagnet);
      // Avoid adding the field on itself twice
      if (otherMagnet != newMagnet)
        newMagnet->addStrayField(otherMagnet);
    }
  }
}

Magnet* MumaxWorld::getMagnet(std::string name) const {
  auto namedMagnet = magnets_.find(name);
  if (namedMagnet == magnets_.end())
    return nullptr;
  return namedMagnet->second;
}

Ferromagnet* MumaxWorld::getFerromagnet(std::string name) const {
    return getMagnetTempl(ferromagnets_, name);
}

Antiferromagnet* MumaxWorld::getAntiferromagnet(std::string name) const {
    return getMagnetTempl(antiferromagnets_, name);
}

Altermagnet* MumaxWorld::getAltermagnet(std::string name) const {
    return getMagnetTempl(altermagnets_, name);
}

NcAfm* MumaxWorld::getNcAfm(std::string name) const {
    return getMagnetTempl(ncafms_, name);
}

const std::map<std::string, Magnet*> MumaxWorld::magnets() const {
  return magnets_;
}

const std::map<std::string, Ferromagnet*> MumaxWorld::ferromagnets() const {
    return getMagnetPointers<Ferromagnet*>(ferromagnets_);
}

const std::map<std::string, Antiferromagnet*> MumaxWorld::antiferromagnets() const {
    return getMagnetPointers<Antiferromagnet*>(antiferromagnets_);
}

const std::map<std::string, Altermagnet*> MumaxWorld::altermagnets() const {
    return getMagnetPointers<Altermagnet*>(altermagnets_);
}

const std::map<std::string, NcAfm*> MumaxWorld::ncafms() const {
    return getMagnetPointers<NcAfm*>(ncafms_);
}

void MumaxWorld::resetTimeSolverEquations(FM_Field torque) const {
  std::vector<DynamicEquation> equations;
  for (const auto& namedMagnet : ferromagnets_) {
    Ferromagnet* magnet = namedMagnet.second.get();
    DynamicEquation eq(
        magnet->magnetization(),
        std::shared_ptr<FieldQuantity>(torque(magnet).clone()),
        std::shared_ptr<FieldQuantity>(thermalNoiseQuantity(magnet).clone()));
    equations.push_back(eq);
  }

  for (const auto& namedMagnet : hostmagnets_) {
    const HostMagnet* magnet = namedMagnet.second;
    for (const Ferromagnet* sub : magnet->sublattices()) {
      DynamicEquation eq(
        sub->magnetization(),
        std::shared_ptr<FieldQuantity>(torque(sub).clone()),
        std::shared_ptr<FieldQuantity>(thermalNoiseQuantity(sub).clone()));
      equations.push_back(eq);
    }
  }

  for (const auto& namedMagnet : magnets_) {
    const Magnet* magnet = namedMagnet.second;

    // add elastodynamics if enabled
    // TODO: this does not play nice with relax()
    if (magnet->enableElastodynamics()) {

      // change in displacement = velocity
      DynamicEquation dvEq(
          magnet->elasticDisplacement(),
          std::shared_ptr<FieldQuantity>(elasticVelocityQuantity(magnet).clone()));
          // No thermal noise
      equations.push_back(dvEq);

      // change in velocity = acceleration
      DynamicEquation vaEq(
          magnet->elasticVelocity(),
          std::shared_ptr<FieldQuantity>(elasticAccelerationQuantity(magnet).clone()));
          // No thermal noise
      equations.push_back(vaEq);
    }
  }

  timesolver_->setEquations(equations);
}

void MumaxWorld::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void MumaxWorld::relax(real tol) {
  Relaxer relaxer(this, this->RelaxTorqueThreshold, tol);
  relaxer.exec();
}


// --------------------------------------------------
// PBC

void MumaxWorld::checkAllMagnetsInMastergrid() const {
  for (const auto& namedMagnet : magnets_) {
    Magnet* magnet = namedMagnet.second;
      if (!inMastergrid(magnet->grid()))
        throw std::out_of_range(
            "Not all magnets of the world fit inside the mastergrid.");
  }
}


void MumaxWorld::recalculateStrayFields() {
  for (const auto& namedMagnet : magnets_) {
    Magnet* magnet = namedMagnet.second;
    for (const auto& magnetStrayField : magnet->strayFields_) {
      StrayField* strayField = magnetStrayField.second;
      strayField->recreateStrayFieldExecutor();
    }
  }
}


// (very) possibly unnecessary
int3 int3min(int3 a, int3 b) {
  int x = std::min(a.x, b.x);
  int y = std::min(a.y, b.y);
  int z = std::min(a.z, b.z);
  return int3{x, y, z};
}
int3 int3max(int3 a, int3 b) {
  int x = std::max(a.x, b.x);
  int y = std::max(a.y, b.y);
  int z = std::max(a.z, b.z);
  return int3{x, y, z};
}

Grid MumaxWorld::boundingGrid() const {
  if (this->magnets_.size() == 0)
    throw std::out_of_range("Cannot find minimum bounding box if there are "
                            "no magnets in the world.");

  if (magnets_.size() == 1)
    return magnets_.begin()->second->grid();

  // if more magnets
  int3 minimum = magnets_.begin()->second->grid().origin();
  int3 maximum = magnets_.begin()->second->grid().origin() +
                 magnets_.begin()->second->grid().size();
  
  auto it = magnets_.begin();  // iterator
  it++;  // go to second magnet
  for (; it != magnets_.end(); it++) {
    Magnet* magnet = it->second;
    minimum = int3min(minimum, magnet->grid().origin());
    maximum = int3max(maximum, magnet->grid().origin() + magnet->grid().size());
  }

  return Grid(maximum - minimum, minimum);  // size, origin
}

void MumaxWorld::setPBC(const int3 pbcRepetitions) {
  checkPbcRepetitions(pbcRepetitions);
  pbcRepetitions_ = pbcRepetitions;

  // find smallest bounding box for all magnets
  Grid mastergrid = boundingGrid();

  // only periodic where specified by user
  int3 size = mastergrid.size();
  if (pbcRepetitions.x == 0)
    size.x = 0;
  if (pbcRepetitions.y == 0)
    size.y = 0;
  if (pbcRepetitions.z == 0)
    size.z = 0;
  mastergrid.setSize(size);

  checkPbcCompatibility(mastergrid, pbcRepetitions);  // should be unnecessary
  mastergrid_ = mastergrid;
  checkAllMagnetsInMastergrid();  // should be unnecessary

  recalculateStrayFields();
}

void MumaxWorld::setPBC(const Grid mastergrid, const int3 pbcRepetitions) {
  checkPbcRepetitions(pbcRepetitions);
  checkPbcCompatibility(mastergrid, pbcRepetitions);
  pbcRepetitions_ = pbcRepetitions;

  mastergrid_ = mastergrid;
  checkAllMagnetsInMastergrid();

  recalculateStrayFields();
}


void MumaxWorld::setPbcRepetitions(int3 pbcRepetitions) {
  checkPbcRepetitions(pbcRepetitions);
  checkPbcCompatibility(mastergrid(), pbcRepetitions);
  pbcRepetitions_ = pbcRepetitions;
  recalculateStrayFields();
}

void MumaxWorld::setMastergrid(Grid mastergrid) {
  checkPbcCompatibility(mastergrid, pbcRepetitions());
  mastergrid_ = mastergrid;
  checkAllMagnetsInMastergrid();
  recalculateStrayFields();
}

void MumaxWorld::unsetPBC() {
  mastergrid_ = Grid(int3{0,0,0});
  pbcRepetitions_ = int3{0,0,0};
  recalculateStrayFields();
}

// --------------------------------------------------
