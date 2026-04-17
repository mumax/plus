#include "altermagnet.hpp"

#include <algorithm>
#include <memory>
#include <math.h>
#include <cfloat>
#include <vector>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "relaxer.hpp"

Altermagnet::Altermagnet(std::shared_ptr<System> system_ptr,
                         std::string name)
    : HostMagnet(system_ptr, name),
      sub1_(Ferromagnet(system_ptr, name + ":sublattice_1", this)),
      sub2_(Ferromagnet(system_ptr, name + ":sublattice_2", this)),
      alterex_1(system(), 0.0, name + ":alterex_1", "J/m"),
      alterex_2(system(), 0.0, name + ":alterex_2", "J/m"),
      alterex_angle(system(), 0.0, name + ":alterex_angle", "rad"),
      interAlterex_1(system(), 0.0, name + ":inter_alterex_1", "J/m"),
      scaleAlterex_1(system(), 1.0, name + ":scale_alterex_1", ""),
      interAlterex_2(system(), 0.0, name + ":inter_alterex_2", "J/m"),
      scaleAlterex_2(system(), 1.0, name + ":scale_alterex_2", "") {
        addSublattice(&sub1_);
        addSublattice(&sub2_);
      }
      
Altermagnet::Altermagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry,
                         GpuBuffer<unsigned int> regions)
    : Altermagnet(std::make_shared<System>(world, grid, geometry, regions), name) {}

const Ferromagnet* Altermagnet::sub1() const {
  return &sub1_;
}

const Ferromagnet* Altermagnet::sub2() const {
  return &sub2_;
}

void Altermagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void Altermagnet::relax(real tol) {
  std::vector<real> threshold = {sub1()->RelaxTorqueThreshold,
                                 sub2()->RelaxTorqueThreshold};
    // If only one sublattice has a user-set threshold, then both
    // sublattices are relaxed using the same threshold.
    if (threshold[0] > 0.0 && threshold[1] <= 0.0)
      threshold[1] = threshold[0];
    else if (threshold[0] <= 0.0 && threshold[1] > 0.0)
      threshold[0] = threshold[1];

    Relaxer relaxer(this, threshold, tol);
    relaxer.exec();
}
