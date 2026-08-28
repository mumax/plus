#include "ncafm.hpp"

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

NcAfm::NcAfm(std::shared_ptr<System> system_ptr,
             std::string name)
    : HostMagnet(system_ptr, name),
      sub1_(Ferromagnet(system_ptr, name + ":sublattice_1", this)),
      sub2_(Ferromagnet(system_ptr, name + ":sublattice_2", this)),
      sub3_(Ferromagnet(system_ptr, name + ":sublattice_3", this)) {
        addSublattice(&sub1_);
        addSublattice(&sub2_);
        addSublattice(&sub3_);
      }
      
NcAfm::NcAfm(MumaxWorld* world,
             Grid grid,
             std::string name,
             GpuBuffer<bool> geometry,
             GpuBuffer<unsigned int> regions)
    : NcAfm(std::make_shared<System>(world, grid, geometry, regions), name) {}

const Ferromagnet* NcAfm::sub1() const {
  return &sub1_;
}

const Ferromagnet* NcAfm::sub2() const {
  return &sub2_;
}

const Ferromagnet* NcAfm::sub3() const {
  return &sub3_;
}

void NcAfm::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void NcAfm::relax(real tol) {
    std::vector<real> threshold = {
        sub1()->RelaxTorqueThreshold,
        sub2()->RelaxTorqueThreshold,
        sub3()->RelaxTorqueThreshold
    };

    auto positive_min = [](real a, real b) -> real {
        if (a > 0.0 && b > 0.0) return std::min(a, b);
        if (a > 0.0) return a;
        if (b > 0.0) return b;
        return -1.0;
    };

    // If only one sublattice has a user-set threshold, then all
    // sublattices are relaxed using the same threshold.
    // If two thresholds are set, propagate the smallest to the remaining one
    for (int i = 0; i < 3; ++i) {
      if (threshold[i] <= 0.0) {
        real a = threshold[(i + 1) % 3];
        real b = threshold[(i + 2) % 3];
        threshold[i] = positive_min(a, b);
      }
    }
    Relaxer relaxer(this, threshold, tol);
    relaxer.exec();
}