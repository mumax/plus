#include "antiferromagnet.hpp"

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

Antiferromagnet::Antiferromagnet(std::shared_ptr<System> system_ptr,
                                 std::string name)
    : HostMagnet(system_ptr, name),
      sub1_(Ferromagnet(system_ptr, name + ":sublattice_1", this)),
      sub2_(Ferromagnet(system_ptr, name + ":sublattice_2", this)) {
        addSublattice(&sub1_);
        addSublattice(&sub2_);
      }
      
Antiferromagnet::Antiferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry,
                         GpuBuffer<unsigned int> regions)
    : Antiferromagnet(std::make_shared<System>(world, grid, geometry, regions), name) {}

const Ferromagnet* Antiferromagnet::sub1() const {
  return &sub1_;
}

const Ferromagnet* Antiferromagnet::sub2() const {
  return &sub2_;
}

void Antiferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void Antiferromagnet::relax(real tol) {
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
